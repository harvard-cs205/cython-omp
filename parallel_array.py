import set_compiler
set_compiler.install()
import pyximport
pyximport.install()

import omp

omp.parallel_array()
