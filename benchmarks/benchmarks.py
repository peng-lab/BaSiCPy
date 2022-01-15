# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import numpy
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve().absolute()))
print(sys.path)
from pybasic.tools import dct2d, idct2d

REPLICATES = 1024


class TimeSuite:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """

    def setup(self):
        self.matrix = numpy.random.rand(1024 ** 2).reshape(1024, 1024)

    def time_dct2(self):
        for _ in range(REPLICATES):
            null = dct2d(self.matrix)

    def time_idct2(self):
        for _ in range(REPLICATES):
            null = idct2d(self.matrix)


class PeakMemSuite:
    def setup(self):
        self.matrix = numpy.random.rand(1024 ** 2).reshape(1024, 1024)

    def peakmem_dct2(self):
        return dct2d(self.matrix)

    def peakmem_idct2(self):
        null = idct2d(self.matrix)
