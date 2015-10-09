from cython.parallel import parallel, prange, threadid
from openmp cimport omp_lock_t, \
    omp_set_lock, omp_unset_lock, \
    omp_init_lock, omp_destroy_lock
cimport openmp
import numpy as np
cimport numpy as np

cpdef parallel_block():
    cdef:
         omp_lock_t lock
         int val

    omp_init_lock(&lock)

    with nogil, parallel(num_threads=100):
        val = threadid()
        omp_set_lock(&lock)
        with gil:
            print("Hello from {}".format(val))
        omp_unset_lock(&lock)

    omp_destroy_lock(&lock)

cpdef parallel_range(int num_threads):
    cdef:
         omp_lock_t lock
         int idx, val

    omp_init_lock(&lock)

    for idx in prange(100, nogil=True, num_threads=num_threads, schedule=dynamic):
        omp_set_lock(&lock)
        val = threadid()
        with gil:
            print("Idx {} thread {}".format(idx, val))
        omp_unset_lock(&lock)

    print("Last: Idx {} thread {}".format(idx, val))
    omp_destroy_lock(&lock)
