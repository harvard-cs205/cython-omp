import set_compiler
set_compiler.install()
import pyximport
pyximport.install(reload_support=True)

import omp
