from numba import njit
import numpy as np
from numba import dppl
from numba.dppl.testing import unittest
from numba.dppl.testing import DPPLTestCase
import dppl.ocldrv as ocldrv


class TestWithDPPLContext(DPPLTestCase):
    def test_with_dppl_context(self):
        @njit
        def nested_func(a, b):
            np.sin(a, b)

        @njit
        def func(b):
            a = np.ones((64), dtype=np.float64)
            nested_func(a, b)

        expected = np.ones((64), dtype=np.float64)
        got_gpu = np.ones((64), dtype=np.float64)
        got_cpu = np.ones((64), dtype=np.float64)

        with ocldrv.igpu_context():
            func(got_gpu)
        with ocldrv.cpu_context():
            func(got_cpu)
        func(expected)

        np.testing.assert_array_equal(expected, got_gpu)
        np.testing.assert_array_equal(expected, got_cpu)


if __name__ == '__main__':
    unittest.main()
