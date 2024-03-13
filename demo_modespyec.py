"""
Script that generates and plots a modespec-like plot for a given DIII-D shot number.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors
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
    parser.add_argument("--nfsmooth", type=int, default=7)
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=8.0)

    parser.add_argument("--eps-int", type=float, default=0.20)
    parser.add_argument("--coh-min", type=float, default=0.925)

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

    # Plotting example helpers

    bbox = [
        spec["tmid"][0],
        spec["tmid"][-1],
        spec["freq"][0] / 1e3,
        spec["freq"][-1] / 1e3,
    ]

    def time_freq_image(thing: np.array, title: str):
        plt.imshow(
            thing,
            origin="lower",
            extent=bbox,
            aspect="auto",
        )
        plt.xlabel("time [sec]")
        plt.ylabel("freq [kHz]")
        plt.title(title)

    def get_coh_min():
        return spec["c95"] if args.coh_min < 0.0 else args.coh_min

    # Time-frequency plots showing internal variables

    time_freq_image(
        np.log10((spec["X11"] + spec["X22"]) / 2.0),
        "Average PSD (%s, %s)" % (probe_name_1, probe_name_2),
    )
    plt.show()

    time_freq_image(
        np.round(-1.0 * (180.0 / np.pi) * np.angle(spec["SX12"]) / delta_theta),
        "cross-phase/delta (%s, %s)" % (probe_name_1, probe_name_2),
    )
    plt.colorbar()
    plt.show()

    time_freq_image(spec["SC12"], "coherence (%s, %s)" % (probe_name_1, probe_name_2))
    plt.colorbar()
    plt.show()

    masked = np.copy(spec["SC12"])
    masked[spec["SC12"] < get_coh_min()] = 0.0
    # masked[spec["SC12"] >= get_coh_min()] = 1.0
    time_freq_image(
        masked, "coherence (%s, %s) > %f" % (probe_name_1, probe_name_2, get_coh_min())
    )
    plt.colorbar()
    plt.show()

    # Create a modespec-like colored spectrogram plot - thresholded by args.coh_min

    modespyec_clist = [
        matplotlib.colors.to_rgb(modespyec.get_color(n)) for n in np.arange(-5, 6, 1)
    ]
    modespyec_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "modespyec", modespyec_clist, N=100
    )
    mpl.colormaps.register(modespyec_cmap)
    time_freq_image(
        modespyec.get_mode_map(spec, delta_theta, get_coh_min(), no_value=0.0),
        "modespyec.get_mode_map",
    )
    plt.gca().set_facecolor("black")
    plt.set_cmap("modespyec")
    plt.clim((-5, 5))
    plt.colorbar(ticks=np.arange(-5, 6, 1))
    plt.show()

    # Extract and plot n-number amplitude traces
    # Use modespec-like coloring scheme

    amps = modespyec.get_amplitude(
        spec,
        [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5],
        delta_theta,
        coh_min=get_coh_min(),
        eps_int=args.eps_int,
    )

    for n in amps.keys():
        plt.plot(spec["tmid"], amps[n], label="n=%i" % (n), c=modespyec.get_color(n))

    plt.legend()
    plt.grid(True)
    plt.gca().set_facecolor("black")
    plt.xlabel("time [sec]")
    plt.ylabel("RMS amplitude [T/s]")
    plt.title("shot #%i" % (args.shot))
    plt.show()

    print("done.")
