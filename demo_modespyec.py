"""
Script that generates and plots a modespec-like plot for a given DIII-D shot number.
"""

import numpy as np
import matplotlib.pyplot as plt
import modespyec 
import argparse

# from toksearch import Pipeline
from toksearch import PtDataSignal

def get_default_probe(shotno : int):
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
    parser.add_argument("--blocksize", type=int, default=500)
    parser.add_argument("--blockstride", type=int, default=250)
    parser.add_argument("--window", type=str, default="Hamming")
    args = parser.parse_args()

    probe_name_1, probe_name_2, delta_theta = get_default_probe(args.shot)

    probe1 = PtDataSignal(probe_name_1).fetch(args.shot)
    probe2 = PtDataSignal(probe_name_2).fetch(args.shot)

    assert np.all(probe1["times"] == probe2["times"])

    time_sec = probe1["times"] * 1.0e-3
    Ts = np.mean(np.diff(time_sec))

    print("shot #%i --> %s, %s" % (args.shot, probe_name_1.upper(), probe_name_2.upper()))
    print("Fs = %.1f kHz" % ( 1.0e-3 / Ts))

    spec = modespyec.wsfft_paired_signal(time_sec, probe1["data"], probe2["data"], args.blocksize, args.blockstride, args.nfft,args.window)

    print("done.")
