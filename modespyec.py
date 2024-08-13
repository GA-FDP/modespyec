import numpy as np


def wsfft_paired_signal(
    t: np.array,
    x1: np.array,
    x2: np.array,
    blocksize: int,
    blockstride: int,
    nfft: int,
    wintype: str,
    nfsmooth: int,
    eps: float,
) -> dict:
    """
    Windowed sliding FFT of two signals x1, x2.
    The analysis window has length = blocksize.
    The FFT has nfft >= blocksize samples.
    Each analysis is separated by blockstride samples.
    The time vector t is assumed to have a constant delta.
    """
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

    # Weight vector used when "integrated" RMS amplitude is requested
    omeghat = np.pi * np.arange(nffth) / nffth
    one_over_omegasq = np.hstack((0.0, 1.0 / omeghat[range(1, nffth)])) ** 2
    assert len(one_over_omegasq) == nffth and len(one_over_omegasq.shape) == 1

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

        if eps > 0:
            SC12[:, b] = np.real(SX12[:, b] * np.conj(SX12[:, b])) / (
                eps + SX11[:, b] * SX22[:, b]
            )
        else:
            SC12[:, b] = (
                np.real(SX12[:, b] * np.conj(SX12[:, b])) / SX11[:, b] / SX22[:, b]
            )

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
        "Ts": Ts,
        "ooomegasq": one_over_omegasq,
        "eps": eps,
    }


def get_mode_map(
    M: dict,
    delta_theta: float,
    integrated: bool = False,
    coh_min: float = None,
    no_value: float = np.nan,
    pfrac: float = None,
    qhigh: float = 0.99,
    tmin: float = None,
    tmax: float = None,
) -> np.array:
    if integrated:
        raise NotImplementedError
    if coh_min is None:
        coh_min = M["c95"]
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
    nnumber: list,
    delta_theta: float,
    nsigned: bool = True,
    integrated: bool = False,
    coh_min: float = 0.98,
    eps_int: float = 0.20,
) -> dict:
    """
    M is the dict returned by the above function.
    nnumber is a list of n-numbers for which to estimate RMS amplitudes
    delta_theta is the angular distance between the probes that was used to calc. M
    coh_min: minimum coherence
    eps_int: tolerance for integerness
    """
    NB = M["tmid"].shape[0]
    NF = M["X11"].shape[0]
    c = 180.0 / np.pi

    def get_single_amplitude_(ntarget):
        W = np.zeros((NF, NB))
        W[M["SC12"] >= coh_min] = 1.0
        if nsigned:
            W[np.abs(-c * np.angle(M["SX12"]) / delta_theta - ntarget) > eps_int] = 0.0
        else:
            W[
                np.abs(np.abs(-c * np.angle(M["SX12"]) / delta_theta) - np.abs(ntarget))
                > eps_int
            ] = 0.0
        return np.sqrt(M["ff"] * np.sum(0.5 * W * (M["X11"] + M["X22"]), axis=0))

    def get_single_integrated_amplitude_(ntarget):
        assert len(M["ooomegasq"]) == NF
        W = np.tile(M["ooomegasq"].reshape((NF, 1)), (1, NB))
        W[M["SC12"] < coh_min] = 0.0
        if nsigned:
            W[np.abs(-c * np.angle(M["SX12"]) / delta_theta - ntarget) > eps_int] = 0.0
        else:
            W[
                np.abs(np.abs(-c * np.angle(M["SX12"]) / delta_theta) - np.abs(ntarget))
                > eps_int
            ] = 0.0
        return M["Ts"] * np.sqrt(
            M["ff"] * np.sum(0.5 * W * (M["X11"] + M["X22"]), axis=0)
        )

    A = dict()
    for n in nnumber:
        A[n] = (
            get_single_integrated_amplitude_(n)
            if integrated
            else get_single_amplitude_(n)
        )

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


def make_default_options() -> dict:
    return {
        "blocksize": 1024,
        "blockstride": 512,
        "nfft": 1024,
        "wintype": "hamming",
        "nfsmooth": 7,
        "abolute-n": True,
        "coh-min": 0.975,
        "eps-int": 0.25,
        "robust-mean-trim-lo": int(2),
        "robust-mean-trim-hi": int(2),
        "amplitude-eps": 0.0,
    }


def summarize_consecutive_circular_pairs(A: dict, options: dict, nlist: list) -> dict:
    def angular_diff(theta0, theta1):
        diff = theta1 - theta0
        return diff if diff > 0 else diff + 360.0

    num_outboard = A["x"].shape[1] - 1
    assert len(A["xnames"]) == num_outboard + 1 and A["xnames"][-1].startswith("mpi11m")

    for i in range(num_outboard - 1):
        assert A["xangle"][i] < A["xangle"][i + 1]

    index_pairs = [(i, (i + 1) % num_outboard) for i in range(num_outboard)]

    theta_diffs = [
        angular_diff(A["xangle"][i0], A["xangle"][i1]) for i0, i1 in index_pairs
    ]

    analyses = [
        wsfft_paired_signal(
            A["t"].flatten(),
            A["x"][:, i0],
            A["x"][:, i1],
            options["blocksize"],
            options["blockstride"],
            options["nfft"],
            options["wintype"],
            options["nfsmooth"],
            options["amplitude-eps"],
        )
        for i0, i1 in index_pairs
    ]

    assert options["robust-mean-trim-lo"] >= 0
    assert options["robust-mean-trim-hi"] >= 0

    rms = list()

    for n in nlist:
        assert n > 0, "positive |n| expected for this call"
        rms_amplitudes_gauss = np.column_stack(
            [
                1.0e4
                * get_amplitude(
                    analysis,
                    [n],
                    theta_diffs[k],
                    nsigned=not options["abolute-n"],
                    integrated=True,
                    coh_min=options["coh-min"],
                    eps_int=options["eps-int"],
                )[n]
                for k, analysis in enumerate(analyses)
            ]
        )

        if options["robust-mean-trim-hi"] > 0:
            robust_mean = np.mean(
                np.sort(rms_amplitudes_gauss, axis=1)[
                    :, options["robust-mean-trim-lo"] : -options["robust-mean-trim-hi"]
                ],
                axis=1,
            )
        else:
            robust_mean = np.mean(
                np.sort(rms_amplitudes_gauss, axis=1)[
                    :, options["robust-mean-trim-lo"] :
                ],
                axis=1,
            )

        rms.append(robust_mean)

    return {
        "index-pairs": index_pairs,
        "theta-diffs": theta_diffs,
        "named-pairs": [(A["xnames"][i0], A["xnames"][i1]) for i0, i1 in index_pairs],
        "nlist": [n for n in nlist],
        "gauss": np.column_stack(rms),
        "time": analyses[0]["tmid"],
    }


def event_detector_(
    t: np.array,
    y: np.array,
    ylevel,
    tdelay,
    breakAtFirst: bool = False,
    initState: int = 0,
    initComp: bool = False,
) -> list:
    """
    ylevel must have two elements: [pos. flank threshold, neg. flank threshold]
    tdelay must have two elements: [pos. flank debounce, neg. flank debounce]
    """
    assert len(ylevel) == 2
    assert len(tdelay) == 2
    tflanks = []
    assert len(t.shape) == 1 and len(y.shape) == 1
    n = len(y)
    assert len(t) == n
    dt = np.diff(t)
    assert initState == 0 or initState == 1
    theState = initState
    isAbove = initComp
    clkLocal = 0.0
    for ii in range(n - 1):
        if breakAtFirst and len(tflanks) > 0:
            break
        clkLocal += dt[ii]
        if not isAbove:
            if y[ii] > ylevel[0]:
                isAbove = True
                clkLocal = 0.0
        else:
            if y[ii] < ylevel[1]:
                isAbove = False
                clkLocal = 0.0
        if theState == 0:
            if isAbove and clkLocal >= tdelay[0]:
                # log a pos. flank @ t(ii) - clkLocal AND switch state to 1
                tflanks.append(t[ii + 1] - clkLocal)
                theState = 1
        else:
            if not isAbove and clkLocal >= tdelay[1]:
                # log a neg. flank @ t(ii) - clkLocal AND switch state to 0
                tflanks.append(t[ii + 1] - clkLocal)
                theState = 0
    return tflanks


def default_detect_events_options() -> dict:
    return {
        "posFlankThreshold": np.linspace(1.0, 10.0, 25),  # [Gauss]
        "whichN": 1,  # |n| number
        "debounceDelayTime": 8.0,  # [msec]
    }


def detect_events(
    B: dict, tstart: float = None, tend: float = None, detect_options: dict = None
) -> dict:
    """
    The input dict B is the returned object from "summarize_consecutive_circular_pairs".
    """

    if detect_options is None:
        detect_options = default_detect_events_options()

    assert len(B["nlist"]) == B["gauss"].shape[1]
    assert len(B["time"]) == B["gauss"].shape[0]

    if tstart is None or (not tstart is None and tstart < B["time"][0]):
        tstart = B["time"][0]

    if tend is None or (not tend is None and tend > B["time"][-1]):
        tend = B["time"][-1]

    col_idx = B["nlist"].index(detect_options["whichN"])
    rms_trace = B["gauss"][:, col_idx]

    num_thresholds = len(detect_options["posFlankThreshold"])
    events = np.repeat(np.nan, num_thresholds)

    time_window = np.logical_and(B["time"] >= tstart, B["time"] <= tend)
    assert np.sum(time_window) > 0
    tdelay = [1.0e-3 * detect_options["debounceDelayTime"] for _ in range(2)]

    for k, thresh in enumerate(detect_options["posFlankThreshold"]):
        ylevel = [thresh for _ in range(2)]
        tflanks = event_detector_(
            B["time"][time_window], rms_trace[time_window], ylevel, tdelay
        )
        if len(tflanks) > 0:
            events[k] = tflanks[0]

    return {
        "tstart": tstart,
        "tend": tend,
        "events": events,
        "gauss_level": detect_options["posFlankThreshold"],
        "debounce_time": detect_options["debounceDelayTime"] * 1.0e-3,  # [sec]
    }
