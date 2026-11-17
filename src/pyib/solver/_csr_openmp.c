/* Optional ordered-row FP64 CSR application. No NumPy C ABI or solver state.
 * The stored order within each row is unchanged; rows alone run in parallel.
 * Bounds are checked inside every application, including aliases modified after
 * Python validation. Inputs must not be mutated concurrently by caller code.
 */
#define _GNU_SOURCE
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <fenv.h>
#include <omp.h>
#include <dlfcn.h>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif
#ifdef __FAST_MATH__
#error "PyIB ordered CSR must not use fast-math"
#endif
#ifndef PYIB_BUILD_FLAGS
#error "Use the PyIB installation build, including its fixed arithmetic flags"
#endif

static int format_is(const Py_buffer *v, char kind, int bytes) {
    const char *f = v->format;
    if (!f || v->itemsize != bytes) return 0;
    if (*f == '@' || *f == '=') ++f;
    return f[0] == kind && f[1] == '\0';
}
static int index_width(const Py_buffer *v) {
    if (format_is(v, 'i', 4) || format_is(v, 'l', 4)) return 4;
    if (format_is(v, 'q', 8) || format_is(v, 'l', 8)) return 8;
    return 0;
}
static int vector_buffer(PyObject *obj, Py_buffer *v, int writable) {
    if (PyObject_GetBuffer(obj, v, PyBUF_FORMAT | PyBUF_STRIDES |
                          (writable ? PyBUF_WRITABLE : 0)) < 0) return 0;
    if (v->ndim != 1 || !v->shape || !v->strides || v->shape[0] < 0 || v->itemsize <= 0 ||
        v->suboffsets != NULL || !PyBuffer_IsContiguous(v, 'C') ||
        (v->shape[0] && (!v->buf || v->strides[0] != v->itemsize)) ||
        v->shape[0] != v->len / v->itemsize || v->len % v->itemsize ||
        (v->len && (uintptr_t)v->buf % (uintptr_t)v->itemsize)) {
        PyErr_SetString(PyExc_ValueError, "Expected aligned native contiguous one-dimensional buffers");
        return 0;
    }
    return 1;
}
static int overlaps(const Py_buffer *a, const Py_buffer *b) {
    if (!a->len || !b->len) return 0;
    uintptr_t x = (uintptr_t)a->buf, y = (uintptr_t)b->buf;
    if ((uintptr_t)a->len > UINTPTR_MAX-x || (uintptr_t)b->len > UINTPTR_MAX-y) return 1;
    return x < y+(uintptr_t)b->len && y < x+(uintptr_t)a->len;
}
static int64_t index_at(const Py_buffer *v, Py_ssize_t i, int width) {
    return width == 4 ? ((const int32_t *)v->buf)[i] : ((const int64_t *)v->buf)[i];
}
static PyObject *apply(PyObject *self, PyObject *args) {
    PyObject *objects[5]; Py_buffer b[5] = {{0}};
    Py_ssize_t rows, cols; int threads, actual_team = 0, error = 0;
    PyObject *answer = NULL;
    if (!PyArg_ParseTuple(args, "nnOOOOOi", &rows, &cols, &objects[0],
                          &objects[1], &objects[2], &objects[3], &objects[4], &threads)) return NULL;
    if (rows < 0 || rows == PY_SSIZE_T_MAX || cols < 0 || threads < 1) {
        PyErr_SetString(PyExc_ValueError, "Invalid dimensions or thread count"); return NULL;
    }
    for (int k=0; k<5; ++k) if (!vector_buffer(objects[k], &b[k], k==4)) goto done;
    int width = index_width(&b[0]);
    if (!width || index_width(&b[1]) != width ||
        !format_is(&b[2], 'd', 8) || !format_is(&b[3], 'd', 8) || !format_is(&b[4], 'd', 8) ||
        b[0].shape[0] != rows+1 || b[1].shape[0] != b[2].shape[0] ||
        b[3].shape[0] != cols || b[4].shape[0] != rows) {
        PyErr_SetString(PyExc_ValueError, "CSR requires native int32/int64 indices and FP64 data/vector/output with matching sizes"); goto done;
    }
    for (int k=0; k<4; ++k) if (overlaps(&b[k], &b[4])) {
        PyErr_SetString(PyExc_ValueError, "Output must not overlap any input buffer"); goto done;
    }
    Py_ssize_t nnz = b[2].shape[0];
    if (index_at(&b[0], 0, width) != 0 || index_at(&b[0], rows, width) != nnz) {
        PyErr_SetString(PyExc_ValueError, "Invalid CSR pointer endpoints"); goto done;
    }
    const double *data=b[2].buf, *x=b[3].buf; double *y=b[4].buf;
    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel num_threads(threads) reduction(|:error)
    {
        int fp_ok = fegetround() == FE_TONEAREST;
        #if defined(__SSE__)
        /* Reject DAZ/FTZ and non-nearest rounding; sticky exception flags are
         * observations of earlier arithmetic, not a change of FP control. */
        fp_ok = fp_ok && ((_mm_getcsr() & 0xe040u) == 0);
        #endif
        if (!fp_ok) error |= 1;
        #pragma omp single
        actual_team = omp_get_num_threads();
        #pragma omp for schedule(static)
        for (Py_ssize_t row=0; row<rows; ++row) {
            int64_t lo=index_at(&b[0],row,width), hi=index_at(&b[0],row+1,width);
            if (lo < 0 || hi < lo || hi > nnz) { error |= 2; continue; }
            double sum=0.;
            for (int64_t j=lo; j<hi; ++j) {
                int64_t col=index_at(&b[1],(Py_ssize_t)j,width);
                if (col < 0 || col >= cols) { error |= 2; break; }
                sum += data[j]*x[col];
            }
            /* Match SciPy's freshly zeroed destination and per-row += sum. */
            y[row] = 0.; y[row] += sum;
            if (!isfinite(y[row])) error |= 4;
        }
    }
    Py_END_ALLOW_THREADS
    if (error || actual_team != threads) {
        PyErr_Format(PyExc_RuntimeError, "OpenMP CSR failed (flags=%d, actual threads=%d, requested=%d); no fallback", error, actual_team, threads);
    } else answer = PyLong_FromLong(actual_team);
done:
    for (int k=0; k<5; ++k) if (b[k].obj) PyBuffer_Release(&b[k]);
    return answer;
}
static PyObject *build_info(PyObject *self, PyObject *unused) {
    Dl_info info;
    if (!dladdr((void *)omp_get_num_threads, &info) || !info.dli_fname) {
        PyErr_SetString(PyExc_RuntimeError, "Cannot identify linked OpenMP runtime"); return NULL;
    }
    return Py_BuildValue("{s:i,s:s,s:s,s:i,s:s}", "abi", 1,
                         "compiler", __VERSION__, "fixed_compile_flags", PYIB_BUILD_FLAGS,
                         "openmp", _OPENMP, "runtime_path", info.dli_fname);
}
static PyMethodDef methods[] = {
    {"apply", apply, METH_VARARGS, "Ordered CSR application into an independent FP64 output buffer."},
    {"build_info", build_info, METH_NOARGS, "Compiler and actual linked OpenMP runtime identity."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_csr_openmp", NULL, -1, methods};
PyMODINIT_FUNC PyInit__csr_openmp(void) { return PyModule_Create(&module); }
