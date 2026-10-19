# SQG ESN reproduction with XESN

This directory contains a self-contained **workflow layer** for the SQG ESN experiments in `tex/main.tex`:

1. `generate_sqg.py` integrates a 64×64, two-level SQG model, spins it up for 360 days, and streams 25 years of 5-minute temperature-anomaly fields to Zarr.
2. `train_esn.py` standardizes from the first 15 years, calibrates the five ESN macro parameters on 10 validation forecasts, then fits and saves the final local readouts.
3. `predict_esn.py` loads a saved XESN model and forecasts from a supplied sequence of fields.

The scripts have **not been run**. They are written against the XESN source tree in `../timothyas-xesn-02ef51b` and the public `sqgturb` API (`from sqgturb import SQG, irfft2`). Install XESN and the upstream SQG model in the chosen Python environment before use. XESN's optional GPU support can accelerate the matrix operations; its Bayesian optimizer is CPU-only.

## Important reproduction details

- The generated Zarr stores raw `theta` in kelvin with dimensions `x,y,z,time`. A sidecar `normalization.json` records one global training mean and standard deviation; XESN applies this affine normalization at load time.
- The first 15 years are training, years 15–20 are macro-validation, and years 20–25 are held-out test data. Forecasts use a 10-day reservoir spin-up and 12-hour lead time.
- `--nsub` selects temporal stride (1, 4, 16, etc.). The model's effective step is `300 * nsub` seconds.
- Readout fitting uses `(R Rᵀ / N + βI) Woutᵀ = (Y Rᵀ / N)ᵀ`, matching the averaged-covariance convention in the manuscript.
- Macro optimization uses manuscript-style global-state NRMSE plus `gamma × KE_NRMSE`. The KE spectrum is calculated with the repository's `rcgfd/xsqgturb.py` diagnostic.
- XESN's internal model format stores `Wout` and the random-matrix construction parameters, not dense `W`/`Win` copies. Keep the exact XESN version and the generated sidecar metadata with each run.

## Typical workflow

Run from this directory after installing dependencies and confirming that `sqgturb` is importable:

```text
python generate_sqg.py --output data/sqg64_300s.zarr
python train_esn.py --data data/sqg64_300s.zarr --nsub 1 --gamma 0.1 --reservoir 6000 --output models/nsub01-gamma0.1-nr6000
python predict_esn.py --model models/nsub01-gamma0.1-nr6000/model.zarr --metadata models/nsub01-gamma0.1-nr6000/run.json --input data/sqg64_300s.zarr --start-index 2076480 --output forecasts/example.zarr
```

For the 12,000-node result discussed in the paper, use `--nsub 16 --gamma 0.1 --reservoir 12000`. The paper's 12,000-node comparison is only reported for that setting. Use separate output directories for each parameter combination.

The training script performs the Bayesian optimization and final readout fit in one invocation. For exact comparison with published plots, retain the fixed seeds, sample indices, Nsub, gamma, and reservoir width for each run. This implementation uses the public upstream SQG integrator rather than the missing `ddc.DataSQGturb` class; numerical differences in integrator versions, initialization, or random-matrix generation can prevent bit-for-bit reproduction.

The forecast input must contain **10 days of history plus the last field at the forecast start time** at the model's time stride. A lone instantaneous field is insufficient to reproduce the paper's ESN spin-up. `--start-index` is the index in the *raw 5-minute Zarr time axis* of the forecast initial field; it must be divisible by `nsub`.

