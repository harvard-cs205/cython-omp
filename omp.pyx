from cython.parallel import parallel, prange, threadid
from openmp cimport omp_set_lock, omp_unset_lock, omp_init_lock, omp_lock_t
cimport openmp
import numpy as np
cimport numpy as np

# Helper functions for locks
cdef void acquire(omp_lock_t *l) nogil:
    omp_set_lock(l)

cdef void release(omp_lock_t *l) nogil:
    omp_unset_lock(l)


cpdef parallel_block():
    cdef int val
    with nogil, parallel(num_threads=100):
        val = threadid()
        with gil:
            print("Hello from {}".format(val))

cpdef parallel_block_with_lock():
    cdef:
         omp_lock_t lock
         int val

    omp_init_lock(&lock)
    with nogil, parallel(num_threads=100):
        val = threadid()
        acquire(&lock)
        with gil:
            print("Hello from {}".format(val))
        release(&lock)
