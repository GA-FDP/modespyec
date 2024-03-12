"""
Script that generates and plots a modespec-like plot for a given DIII-D shot number.
"""

import numpy as np
import matplotlib.pyplot as plt
import modespyec
import argparse

from toksearch import PtDataSignal


def get_default_probe(shotno: int):
    if shotno >= 152000:
        return "mpi66m307d", "mpi66m340d", 33.0
    elif shotno >= 144783:
        return "mpi66m307e", "mpi66m340e", 33.0
    elif shotno >= 144760:
        return "mpi66m307d", "mpi66m340d", 33.0
    else:
        raise NotImplementedError


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--shot", type=int, default=150792)
    parser.add_argument("--nfft", type=int, default=1024)
    parser.add_argument("--blocksize", type=int, default=800)  # 4ms window
    parser.add_argument("--blockstride", type=int, default=400)  # 2ms stride
    parser.add_argument("--window", type=str, default="Hamming")
    parser.add_argument("--nfsmooth", type=int, default=5)
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=8.0)
    args = parser.parse_args()

    probe_name_1, probe_name_2, delta_theta = get_default_probe(args.shot)

    probe1 = PtDataSignal(probe_name_1).fetch(args.shot)
    probe2 = PtDataSignal(probe_name_2).fetch(args.shot)

    assert np.all(probe1["times"] == probe2["times"])

    time_sec = probe1["times"] * 1.0e-3
    Ts = np.mean(np.diff(time_sec))

    print(
        "shot #%i --> %s, %s" % (args.shot, probe_name_1.upper(), probe_name_2.upper())
    )
    print("Fs = %.1f kHz" % (1.0e-3 / Ts))

    range_filter = np.logical_and(time_sec >= args.tmin, time_sec <= args.tmax)

    spec = modespyec.wsfft_paired_signal(
        time_sec[range_filter],
        probe1["data"][range_filter],
        probe2["data"][range_filter],
        args.blocksize,
        args.blockstride,
        args.nfft,
        args.window,
        args.nfsmooth,
    )

    # Plotting examples

    bbox = [
        spec["tmid"][0],
        spec["tmid"][-1],
        spec["freq"][0] / 1e3,
        spec["freq"][-1] / 1e3,
    ]
    plt.imshow(
        np.log10((spec["X11"] + spec["X22"]) / 2.0),
        origin="lower",
        extent=bbox,
        aspect="auto",
    )
    plt.xlabel("time [sec]")
    plt.ylabel("freq [kHz]")
    plt.title("Average PSD (%s, %s)" % (probe_name_1, probe_name_2))
    plt.show()

    plt.imshow(
        -1.0 * (180.0 / np.pi) * np.angle(spec["SX12"]) / delta_theta,
        origin="lower",
        extent=bbox,
        aspect="auto",
    )
    plt.colorbar()
    plt.xlabel("time [sec]")
    plt.ylabel("freq [kHz]")
    plt.title("cross-phase (%s, %s)" % (probe_name_1, probe_name_2))
    plt.show()

    plt.imshow(
        spec["SC12"],
        origin="lower",
        extent=bbox,
        aspect="auto",
    )
    plt.colorbar()
    plt.xlabel("time [sec]")
    plt.ylabel("freq [kHz]")
    plt.title("coherence (%s, %s)" % (probe_name_1, probe_name_2))
    plt.show()

    print("done.")
