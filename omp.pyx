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


cpdef parallel_reduction():
    cdef:
         int idx, sum = 0
    for idx in prange(100, nogil=True, num_threads=10):
        sum += idx
    print("Sum {}".format(sum))


cpdef nested_parallel_prange():
    cdef:
         int idx, idx2, val
         omp_lock_t lock
    omp_init_lock(&lock)

    with nogil, parallel(num_threads=10):
        with gil:
            print ("outer")
        for idx in prange(100):
            for idx2 in prange(5):
                val = threadid()
                omp_set_lock(&lock)

                with gil:
                    if idx % 10 == 0:
                        print("idx {} idx2 {} thread {}".format(idx, idx2, val))
                omp_unset_lock(&lock)
