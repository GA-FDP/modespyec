# modespyec

Vanilla `modespec`-like spectrogram capabilities. Requires only `numpy`. The demonstration script also requires `matplotlib` and `toksearch`.

## Usage
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
