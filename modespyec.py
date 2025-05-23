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
    nfsmooth: int,
) -> dict:
    assert len(t.shape) == 1
    assert len(x1.shape) == 1
    assert len(x2.shape) == 1
    Ts = np.mean(np.diff(t))
    Fs = 1.0 / Ts
    assert Ts > 0
    assert nfft >= blocksize
    N = x1.shape[0]
    assert x2.shape[0] == N and t.shape[0] == N
    block_start = np.arange(0, N - blocksize, blockstride)
    NB = len(block_start)
    assert nfft % 2 == 0, "please use even number for nfft"
    nffth = nfft // 2
    freq = (np.arange(nffth) / nffth) * (Fs / 2.0)
    w = get_window_weights(wintype, blocksize)
    pw = np.sum(w * w) / blocksize

    # Factor ff is used like this: RMS(.) = sqrt( ff * sum_over_freq (windowed_fft_vector) )
    ff = 2 / (pw * blocksize * nfft)

    assert nfsmooth > 2, "nfsmooth must be > 2"
    assert (nfsmooth - 1) % 2 == 0, "please use odd number for nfsmooth"

    T = np.zeros((NB,))
    X11 = np.zeros((nffth, NB))
    X22 = np.zeros((nffth, NB))
    X12 = np.zeros((nffth, NB), dtype=complex)  # cross-spectrum
    SX11 = np.copy(X11)
    SX22 = np.copy(X22)
    SX12 = np.copy(X12)
    SC12 = np.copy(X11)  # real-valued
    for b in range(NB):
        i1 = block_start[b]
        i2 = i1 + blocksize
        assert i2 < N
        T[b] = (t[i1] + t[i2]) / 2

        idxb = np.arange(i1, i2)

        x1b = x1[idxb] - np.mean(x1[idxb])
        X1b = np.fft.fft(w * x1b, n=nfft)[:nffth]
        x2b = x2[idxb] - np.mean(x2[idxb])
        X2b = np.fft.fft(w * x2b, n=nfft)[:nffth]

        X11[:, b] = np.real(X1b * np.conj(X1b))
        X22[:, b] = np.real(X2b * np.conj(X2b))
        X12[:, b] = X1b * np.conj(X2b)

        SX11[:, b] = smooth(X11[:, b], nfsmooth)
        SX22[:, b] = smooth(X22[:, b], nfsmooth)
        SX12[:, b] = smooth(X12[:, b], nfsmooth)

        SC12[:, b] = np.real(SX12[:, b] * np.conj(SX12[:, b])) / SX11[:, b] / SX22[:, b]

    # modespec's relationship btw. "nsmooth" and "fsmooth"
    df = 1.0 / (blocksize * Ts)
    fsmooth = 2 * df * ((nfsmooth - 1) / 2)

    # modespec's calc (verbatim) of the "c95" value (used as coherence threshold)
    c95 = 1.96 / np.sqrt(2.0 * nfsmooth - 2.0)
    c95 = (np.exp(c95) - np.exp(-c95)) / (np.exp(c95) + np.exp(-c95))
    c95 = c95 * c95

    return {
        "tmid": T,
        "freq": freq,
        "X11": X11,
        "X22": X22,
        "X12": X12,
        "SX11": SX11,
        "SX22": SX22,
        "SX12": SX12,
        "SC12": SC12,
        "deltat": blocksize * Ts,
        "deltaf": df,
        "fsmooth": fsmooth,
        "ff": ff,
        "c95": c95,
    }


def get_mode_map(
    M: dict,
    delta_theta: float,
    coh_min: float,
    no_value: float = np.nan,
    pfrac: float = None,
    qhigh: float = 0.99,
    tmin: float = None,
    tmax: float = None,
) -> np.array:
    nmap = -1 * (180 / np.pi) * np.angle(M["SX12"]) / delta_theta
    nmap[M["SC12"] < coh_min] = no_value
    if not pfrac is None:
        assert pfrac > 0.0 and pfrac <= 1.0
        P = (M["X11"] + M["X22"]) / 2
        if tmin is None:
            tmin = M["tmid"][0]
        if tmax is None:
            tmax = M["tmid"][-1]
        quantile_time_window = np.logical_and(M["tmid"] >= tmin, M["tmid"] <= tmax)
        assert (
            np.sum(quantile_time_window) > 0
        ), "empty time window for peak power reference value"
        phigh = np.quantile(P[:, quantile_time_window], q=qhigh)
        pmin = pfrac * phigh
        nmap[P < pmin] = no_value
    return nmap


def get_amplitude(
    M: dict,
    nsigned: list,
    delta_theta: float,
    coh_min: float = 0.98,
    eps_int: float = 0.20,
) -> dict:
    """
    M is the dict returned by the above function.
    nsigned is a list of n-numbers for which to estimate RMS amplitudes
    delta_theta is the angular distance between the probes that was used to calc. M
    coh_min: minimum coherence
    eps_int: tolerance for integerness
    """

    def get_single_amplitude_(ntarget):
        NB = M["tmid"].shape[0]
        NF = M["X11"].shape[0]
        c = 180.0 / np.pi
        W = np.zeros((NF, NB))
        W[M["SC12"] >= coh_min] = 1.0
        W[np.abs(-c * np.angle(M["SX12"]) / delta_theta - ntarget) > eps_int] = 0.0
        return np.sqrt(M["ff"] * np.sum(0.5 * W * (M["X11"] + M["X22"]), axis=0))

    A = dict()
    for n in nsigned:
        A[n] = get_single_amplitude_(n)

    return A


def get_color(n: int) -> str:
    """
    Get CSS color identifier string (working for matplotlib) corresponding
    to modespec's coloring scheme for n-numbers; return grey if n is out-of-range.
    SEE: https://matplotlib.org/stable/gallery/color/named_colors.html
    """
    modespec_coloring = {
        0: "black",
        1: "red",
        2: "yellow",
        3: "green",
        4: "blue",
        5: "orange",
        -1: "deepskyblue",
        -2: "khaki",
        -3: "violet",
        -4: "cyan",
        -5: "lightgreen",
    }
    return modespec_coloring.get(n, "grey")


def get_window_weights(name: str, blocksize: int) -> np.array:
    nvec = np.arange(blocksize)
    lname = name.lower()
    if lname == "hann":
        return 0.5 * (1 - np.cos(2 * np.pi * nvec / blocksize))
    elif lname == "hamming":
        a1 = 0.5400
        b1 = 1 - a1
        return a1 - b1 * np.cos(2 * np.pi * nvec / blocksize)
    elif lname == "bh":
        a = 2 * np.pi * nvec / blocksize
        return (
            0.35875
            - 0.48829 * np.cos(a)
            + 0.14128 * np.cos(2.0 * a)
            - 0.01168 * np.cos(3.0 * a)
        )
    elif lname == "boxcar":
        return np.ones((blocksize,))
    else:
        raise NotImplementedError


def smooth(x: np.array, npts: int) -> np.array:
    """
    Basic boxcar 1D signal smoothing
    """
    assert len(x.shape) == 1
    nh = (npts - 1) // 2
    assert nh >= 1
    y = np.convolve(x, np.ones((npts,)) / npts, mode="full")
    return y[nh:-nh]
