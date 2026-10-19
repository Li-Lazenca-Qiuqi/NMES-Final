"""Forecast with a trained XESN model from a raw SQG field trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr


def open_data(path: Path) -> xr.Dataset:
    if path.is_dir() or path.suffix == ".zarr":
        return xr.open_zarr(str(path))
    return xr.open_dataset(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="XESN model.zarr")
    parser.add_argument("--metadata", type=Path, required=True, help="run.json from train_esn.py")
    parser.add_argument("--input", type=Path, required=True, help="Raw theta Zarr or NetCDF trajectory")
    parser.add_argument("--output", type=Path, required=True, help="Forecast Zarr output")
    parser.add_argument("--start-index", type=int, required=True,
                        help="Forecast initial index on the raw 5-minute input time axis")
    parser.add_argument("--steps", type=int, default=None, help="Forecast steps; default is 12 hours")
    parser.add_argument("--spinup-steps", type=int, default=None,
                        help="Reservoir spin-up steps; default is 10 days at this model Nsub")
    parser.add_argument("--nsub", type=int, default=None,
                        help="Temporal stride in the input; default comes from run.json")
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    nsub = int(args.nsub if args.nsub is not None else metadata["nsub"])
    if args.start_index % nsub:
        parser.error("--start-index must be divisible by nsub")
    n_spinup = int(args.spinup_steps if args.spinup_steps is not None else metadata["n_spinup"])
    n_steps = int(args.steps if args.steps is not None else metadata["n_steps_12h"])
    if args.start_index < n_spinup * nsub:
        parser.error("Not enough preceding raw samples for the requested reservoir spin-up")

    try:
        from dask.distributed import Client
        import xesn  # noqa: F401
        from xesn import from_zarr
    except ImportError as exc:
        raise ImportError("Install XESN and Dask in the selected environment before forecasting.") from exc

    source = open_data(args.input)
    if metadata.get("field_name", "theta") not in source:
        raise KeyError(f"Field {metadata.get('field_name', 'theta')!r} not found in {args.input}")
    raw = source[metadata.get("field_name", "theta")]
    raw = raw.transpose("x", "y", "z", "time")
    if nsub > 1:
        raw = raw.isel(time=slice(None, None, nsub))
    forecast_index = args.start_index // nsub
    #history = raw.isel(time=slice(forecast_index - n_spinup, forecast_index + 1))
    # 为了让 XESN 内部算出正确长度的 ftime，必须把未来的真实时间点也切进来
    history = raw.isel(time=slice(forecast_index - n_spinup, forecast_index + n_steps + 1))

    if history.sizes["time"] < n_spinup + 1:
        raise ValueError(
            f"Need {n_spinup + 1} fields (10-day spin-up plus initial state); got {history.sizes['time']}."
        )

    norm = metadata["normalization"]
    history = (history - float(norm["bias"])) / float(norm["scale"])
    output_path = args.output
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite forecast output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Client():
        esn = from_zarr(str(args.model))
        prediction = esn.predict(history, n_steps=n_steps, n_spinup=n_spinup)
        prediction = prediction * float(norm["scale"]) + float(norm["bias"])
        prediction.name = "prediction"
        prediction.attrs.update({"units": "K", "description": "XESN SQG temperature-anomaly forecast"})
        prediction.to_dataset().to_zarr(str(output_path), mode="w", zarr_format=2)
    print(f"Saved {n_steps} forecast steps to {output_path}")


if __name__ == "__main__":
    main()

