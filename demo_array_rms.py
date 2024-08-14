"""
Run modespyec for a specific list of shots 
and visualize the amplitude traces (for |n|=1,2,3).

EXAMPLES:
  python -i demo_array_rms.py --shots 159299 162935 195068 193280

"""

import matplotlib.pyplot as plt

import argparse
import numpy as np
import toksearch

import modespyec

import time


def create_signal_dict(rec, signames):
    ncol = len(signames)
    # determine how many samples there are for each named signal
    nsamp = np.zeros((ncol, 1))
    tsamp = np.zeros((ncol, 1))
    for k in range(len(signames)):
        if rec[signames[k]] is not None:
            nsamp[k] = len(rec[signames[k]]["data"])
            assert nsamp[k] == len(rec[signames[k]]["times"])
            tsamp[k] = np.mean(np.diff(rec[signames[k]]["times"])) * 1.0e-3
    nrow = int(
        np.max(nsamp)
    )  # also need to pick the timebase from the argmax index ...
    if nrow == 0:
        return None
    hasX = np.tile(False, (ncol,))
    X = np.tile(np.nan, (nrow, ncol))
    t = np.tile(np.nan, (nrow, 1))
    for k in range(len(signames)):
        if len(rec[signames[k]]["data"]) == nrow:
            X[:, k] = rec[signames[k]]["data"]
            if np.isnan(t[0, 0]):  # only need to copy this once
                t[:, 0] = rec[signames[k]]["times"] * 1.0e-3
        hasX[k] = not np.isnan(X[0, k])
    Ts = np.mean(np.diff(t.transpose()))
    assert len(hasX) == ncol
    return {
        "x": X,
        "t": t,
        "shot": rec["shot"],
        "xnames": signames,
        "hasx": hasX,
        "Ts": Ts,
    }


def update_signal_dict_with_angles(D, angle_dict, makeNumpyArray=True):
    if "xnames" in D.keys():
        xangle = []
        for xname in D["xnames"]:
            xangle.append(angle_dict[xname])
        if makeNumpyArray:
            D.update({"xangle": np.array(xangle)})
        else:
            D.update({"xangle": xangle})


def define_names_and_angles():
    midplane_magnetic_ptnames = [
        "mpi66m067",
        "mpi66m097",
        "mpi66m127",
        "mpi66m157",
        "mpi66m247",
        "mpi66m277",
        "mpi66m307",
        "mpi66m322",
        "mpi66m340",
        "mpi11m322",
    ]

    midplane_toroidal_angles = [
        67.5,
        97.4,
        127.9,
        157.6,
        246.4,
        277.5,
        307.0,
        317.4,
        339.7,
        322.5,
    ]

    assert len(midplane_magnetic_ptnames) == len(midplane_toroidal_angles)

    mag_suffix = "d"
    mag_probe_angle = dict()
    for k in range(len(midplane_toroidal_angles)):
        mag_probe_angle.update(
            {midplane_magnetic_ptnames[k]: midplane_toroidal_angles[k]}
        )

    return midplane_magnetic_ptnames, mag_probe_angle, mag_suffix


def create_pipeline(
    shots,
    ptdata_ptnames,
    mag_suffix,
    angle_dict,
    theNlist=[1, 2, 3],
    keep_source_signals=False,
    keep_prepared_signals=False,
):
    pipe = toksearch.Pipeline(shots)

    for ptname in ptdata_ptnames:
        pipe.fetch(ptname, toksearch.PtDataSignal(ptname + mag_suffix))

    @pipe.map
    def prep_signals(rec):
        A = create_signal_dict(rec, ptdata_ptnames)
        update_signal_dict_with_angles(A, angle_dict)
        rec["A"] = A

    @pipe.map
    def calc_modespyec(rec):
        if rec["A"] is not None:
            analysis_options = modespyec.make_default_options()
            tic0 = time.time()
            report = modespyec.summarize_consecutive_circular_pairs(
                rec["A"], analysis_options, theNlist
            )
            tic1 = time.time()
            rec["options-modespyec"] = analysis_options
            rec["report-modespyec"] = report
            rec["elapsed-modespyec"] = tic1 - tic0

    @pipe.map
    def cleanup(rec):
        if not keep_prepared_signals:
            rec.pop("A")
        if not keep_source_signals:
            for ptname in ptdata_ptnames:
                rec.pop(ptname)

    return pipe


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--shots", type=int, nargs="+")

    args = parser.parse_args()

    names, angles, suffix = define_names_and_angles()

    if args.shots is None:
        print("no shot specified, provide --shots")
        quit()

    assert isinstance(args.shots, list) and len(args.shots) > 0

    pipe = create_pipeline(
        args.shots,
        names,
        suffix,
        angles,
        keep_source_signals=False,
        keep_prepared_signals=True,
    )

    print(args.shots)

    start_time = time.time()

    results = pipe.compute_serial()

    elapsed_time = time.time() - start_time
    print("total serial execution: %.2f sec" % (elapsed_time))

    elapsed_modespyec_calcs = [
        r["elapsed-modespyec"] for r in filter(lambda z: len(z["errors"]) == 0, results)
    ]
    print("(modespyec calc: %.2f sec)" % (np.sum(elapsed_modespyec_calcs)))

    for rec in results:
        if len(rec["errors"]) > 0:
            print(
                "skipping shot #%i (error count = %i)"
                % (rec["shot"], len(rec["errors"]))
            )
            continue

        tvec_new = rec["report-modespyec"]["time"]
        # amp1_new = rec["report-modespyec"]["gauss"][:, 0]
        # amp2_new = rec["report-modespyec"]["gauss"][:, 1]
        # amp3_new = rec["report-modespyec"]["gauss"][:, 2]

        modespyec_label = [
            (
                (
                    "[modespyec] |n|=%i"
                    if rec["options-modespyec"]["abolute-n"]
                    else "[modespyec] n=%i"
                )
                % (n)
            )
            for n in [1, 2, 3]
        ]

        colors = ["red", "orange", "purple"]

        for k in range(3):
            plt.plot(
                tvec_new,
                rec["report-modespyec"]["gauss"][:, k],
                label=modespyec_label[k],
                lw=2,
                alpha=0.75,
                c=colors[k],
            )

        plt.xlabel("time [sec]")
        plt.grid(True)
        plt.legend()
        plt.ylabel("Amplitude [T]")
        plt.title("#%i" % (rec["shot"]))
        plt.show()

    print("done.")
