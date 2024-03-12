import numpy as np

"""
Windowed sliding FFT of two signals x1, x2. 
The analysis window has length = blocksize.
The FFT has nfft >= blocksize samples. 
Each analysis is separated by blockstride samples.
The time vector t is assumed to have a constant delta.
"""


def wsfft_paired_signal(
    t: np.array,
    x1: np.array,
    x2: np.array,
    blocksize: int,
    blockstride: int,
    nfft: int,
    wintype: str,
) -> dict:
    assert len(t.shape) == 1
    assert len(x1.shape) == 1
    assert len(x2.shape) == 1
    Ts = np.mean(np.diff(t))
    assert Ts > 0
    assert nfft >= blocksize
    N = x1.shape[0]
    assert x2.shape[0] == N and t.shape[0] == N
    block_start = np.arange(0, N - blocksize, blockstride)
    NB = len(block_start)
    for b in range(NB):
        i1 = block_start[b]
        i2 = i1 + blocksize
        assert i2 < N
        # ...
        # print([i1,i2])

    return {}


def modespec(WSFFT: dict):
    raise NotImplementedError
