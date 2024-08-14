# modespyec

Vanilla `modespec`-like spectrogram capabilities. Requires only `numpy`. The demonstration script also requires `matplotlib` and `toksearch`.

## Basic usage
```
import modespyec
...
```

See `demo_modespyec.py` for details. To generate a set of demonstration plots simply run the script as follows.
```
python demo_modespec.py
```
Click through the plots one at a time. First a few internal variables are shown. Then a colored `modespec`-like spectrogram is shown, and finally a `modespec`-like mode amplitude traces plot is generated.

The following example includes both TM modes and broadband turbulent activity
```
python demo_modespyec.py --shot 163518 --coh-min 0.40  --nfsmooth 11 --eps-int 0.50
```

Another example:
```
python demo_modespyec.py --shot 169856 --tmax 5.0 --coh-min -1 --pow-frac 0.33 --nfsmooth 5 --window boxcar --eps-int 0.50 --only-modespec-plot
```

## Using entire midplane array to extract amplitude traces

See demonstration script `demo_array_rms.py`. 

Specific example call:
```
python demo_array_rms.py --shots 159299 162935 195068 193280
```

A pipeline is setup that fetches all the midplane fluctuation array probes. The pipeline applies pairwise analysis to consecutive probes along the midplane circle. It then combines all results to a RMS amplitude trace for a set of toroidal mode numbers $|n|=1,2,3$.
 