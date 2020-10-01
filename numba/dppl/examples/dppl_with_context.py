import numpy as np
from numba import dppl, njit, prange
import dpctl
import dpctl.ocldrv as ocldrv


@njit
def g(a):
    return a + 1


@njit
def f(a, b, c, N):
    for i in prange(N):
        a[i] = b[i] + g(c[i])


def main():
    N = 10
    a = np.ones(N)
    b = np.ones(N)
    c = np.ones(N)

    if ocldrv.has_gpu_device:
        with dpctl.device_context(dpctl.device_type.gpu):
            f(a, b, c, N)
    elif ocldrv.has_cpu_device:
        with dpctl.device_context(dpctl.device_type.cpu):
            f(a, b, c, N)
    else:
        print("No device found")


if __name__ == '__main__':
    main()
