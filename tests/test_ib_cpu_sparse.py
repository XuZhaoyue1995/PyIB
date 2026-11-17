"""CPU CSR safety/parity and durable backend identity tests.

Execute on a compute node in an installed PyIB environment. Extension tests
skip on a normal pure-Python install; PYIB_REQUIRE_OPENMP_TESTS=1 makes absence
a failure for the optional-build verification job. No local CI assumptions.
"""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix

from ib_cpu_sparse import OpenMPCSR, _extension, execution_identity, validate_options


class SelectionTests(unittest.TestCase):
    def test_defaults_do_not_require_extension(self):
        from ib_case import validate_config
        config = validate_config({})
        self.assertEqual(config["solver"]["cpu_sparse_backend"], "scipy")
        self.assertEqual(config["solver"]["cpu_sparse_threads"], 1)
        with patch("ib_cpu_sparse._extension", side_effect=AssertionError("loaded optional extension")):
            self.assertEqual(execution_identity(config)["backend"], "scipy")
            cuda = copy.deepcopy(config); cuda["runtime"]["device"] = "cuda"
            self.assertEqual(execution_identity(cuda)["backend"], "cupy")
            config["solver"]["sparse"] = False
            self.assertEqual(execution_identity(config)["backend"], "matrix_free")

    def test_explicit_missing_extension_does_not_fall_back(self):
        from ib_case import validate_config
        config = validate_config({"solver": {"cpu_sparse_backend": "openmp"}})
        with patch("ib_cpu_sparse.importlib.import_module", side_effect=ImportError("not installed")):
            with self.assertRaisesRegex(RuntimeError, "No runtime build or fallback"):
                execution_identity(config)

    def test_invalid_options(self):
        for backend, threads, extra in [("auto",1,{}),([],1,{}),("scipy",4,{}),
                ("openmp",True,{}),("openmp",0,{}),("openmp",1.5,{}),
                ("openmp",1,{"device":"cuda"}),("openmp",1,{"precision":"float32"}),
                ("openmp",1,{"sparse":False})]:
            with self.subTest(backend=backend, threads=threads, extra=extra):
                with self.assertRaises(ValueError): validate_options(backend,threads,**extra)

    def test_checkpoint_rejects_changed_execution_identity_before_state_restore(self):
        from ib_case import (validate_config, config_fingerprint, load_checkpoint, CaseConfigError,
                             CHECKPOINT_SCHEMA_VERSION)
        from validate_ib_result import CHECKPOINT_FIELDS
        config=validate_config({})
        identity=execution_identity(config)
        runtime=SimpleNamespace(config=config,dt=config["time"]["dt"],layout_fingerprint="a"*64,
                                source_hashes={},sparse_execution=identity)
        payload={name:np.int64(0) for name in CHECKPOINT_FIELDS}
        payload.update(checkpoint_schema=np.int64(CHECKPOINT_SCHEMA_VERSION),update_phase=np.str_("post_step"),
                       config_fingerprint=np.str_(config_fingerprint(config)),
                       layout_fingerprint=np.str_(runtime.layout_fingerprint),
                       dt=np.float64(runtime.dt),previous_dt=np.float64(runtime.dt),
                       precision=np.str_("float64"),device=np.str_("cpu"),
                       source_hashes_json=np.str_("{}"))
        changed_binary=copy.deepcopy(identity)
        changed_binary["implementation_sha256"]["scipy_extension"]="0"*64
        changed_backend=copy.deepcopy(identity);changed_backend["backend"]="openmp"
        changed_threads=copy.deepcopy(identity);changed_threads["threads"]=2
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"checkpoint.npz"
            for changed in (changed_binary,changed_backend,changed_threads):
                with self.subTest(changed=changed):
                    payload["sparse_execution_json"]=np.str_(json.dumps(changed,sort_keys=True,separators=(",",":")))
                    np.savez(path,**payload)
                    # No core/driver exists on this sentinel runtime. Success
                    # means rejection preceded any state reconstruction.
                    with self.assertRaisesRegex(CaseConfigError,"sparse execution identity differs"):
                        load_checkpoint(path,runtime)


class CompiledCSRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try: cls.kernel = _extension()
        except RuntimeError:
            if os.environ.get("PYIB_REQUIRE_OPENMP_TESTS") == "1": raise
            raise unittest.SkipTest("Optional OpenMP extension not installed")

    def apply(self, ip, ix, data, x, rows=3, cols=4, threads=1, out=None):
        if out is None: out=np.zeros(rows,dtype=np.float64)
        team=self.kernel.apply(rows,cols,ip,ix,data,x,out,threads)
        self.assertEqual(team,threads)
        return out

    def test_rectangular_unsorted_empty_rows_and_thread_parity(self):
        # Unsorted entries and cancellation exercise stored-order accumulation;
        # an empty row must not inherit uninitialized output.
        for kind in (np.int32,np.int64):
            ip=np.array([0,3,3,5],dtype=kind);ix=np.array([2,0,1,3,0],dtype=kind)
            data=np.array([1e16,1.,-1e16,2.,-3.]);x=np.array([1.,1.,1.,-2.])
            matrix=csr_matrix((data,ix,ip),shape=(3,4))
            matrix.indices=ix;matrix.indptr=ip  # preserve int64 test ABI
            native=matrix@x
            controls=[]
            for threads in (1,2,4):
                operator=OpenMPCSR(matrix,threads)
                actual=operator@x
                np.testing.assert_array_equal(actual.view(np.uint64),native.view(np.uint64))
                self.assertEqual(operator.last_actual_threads,threads)
                controls.append(actual)
            np.testing.assert_array_equal(controls[0].view(np.uint64),controls[-1].view(np.uint64))

    def test_zero_nnz_and_zero_dimensions(self):
        for rows,cols in [(3,4),(0,4),(3,0),(0,0)]:
            for kind in (np.int32,np.int64):
                actual=self.apply(np.zeros(rows+1,dtype=kind),np.empty(0,dtype=kind),
                                  np.empty(0),np.zeros(cols),rows,cols,2)
                np.testing.assert_array_equal(actual,np.zeros(rows))

    def test_low_level_invalid_csr_cannot_access_outside_buffers(self):
        good=(np.array([0,1,1,2],np.int32),np.array([0,3],np.int32),np.ones(2),np.ones(4))
        cases=[(np.array([1,1,1,2],np.int32),good[1],good[2],good[3]),
               (np.array([0,2,1,2],np.int32),good[1],good[2],good[3]),
               (np.array([0,-1,1,2],np.int32),good[1],good[2],good[3]),
               (np.array([0,3,3,2],np.int32),good[1],good[2],good[3]),
               (good[0],np.array([0,4],np.int32),good[2],good[3]),
               (good[0],np.array([-1,3],np.int32),good[2],good[3]),
               (good[0].astype(np.int64),good[1],good[2],good[3])]
        for args in cases:
            with self.subTest(args=args):
                with self.assertRaises((ValueError,RuntimeError)): self.apply(*args)
        # Mutating an index through a retained alias after construction is also
        # checked by C. No Python readonly flag is the safety boundary.
        matrix=csr_matrix((good[2],good[1],good[0]),shape=(3,4))
        operator=OpenMPCSR(matrix,1);matrix.indices[-1]=999999999
        with self.assertRaises(RuntimeError): operator@good[3]

    def test_buffers_endian_alignment_shape_stride_alias_and_readonly(self):
        ip=np.array([0,1,1,2],np.int32);ix=np.array([0,3],np.int32);data=np.ones(2);x=np.ones(4)
        nonnative = ">" if np.little_endian else "<"
        for a,b,c,d in [(ip.astype(nonnative+'i4'),ix,data,x),
                        (ip,ix,data.astype(nonnative+'f8'),x),
                        (ip,ix,data,x.astype(np.float32)),
                        (ip,ix,data,x.reshape(2,2)),
                        (ip,ix,data,np.ones(8)[::2])]:
            with self.assertRaises((ValueError,BufferError)): self.apply(a,b,c,d)
        unaligned=np.ndarray((4,),dtype=np.float64,buffer=bytearray(33),offset=1)
        with self.assertRaises(ValueError): self.apply(ip,ix,data,unaligned)
        with self.assertRaises(ValueError): self.apply(ip,ix,data,x,out=x[:3])
        out=np.zeros(3);out.flags.writeable=False
        with self.assertRaises((ValueError,BufferError)): self.apply(ip,ix,data,x,out=out)
        with self.assertRaises(RuntimeError): self.apply(ip,ix,np.array([np.nan,1.]),x)

    def test_identity_records_actual_binary_and_team(self):
        from ib_case import validate_config
        config=validate_config({"solver":{"cpu_sparse_backend":"openmp","cpu_sparse_threads":2}})
        observed=execution_identity(config,check_team=True)
        self.assertEqual(observed,execution_identity(config))
        self.assertEqual(observed["actual_threads"],2)
        self.assertEqual(set(observed["implementation_sha256"]),{"extension","openmp_runtime"})
        self.assertIn("-ffp-contract=off",observed["build"]["fixed_compile_flags"])


if __name__ == "__main__": unittest.main()
