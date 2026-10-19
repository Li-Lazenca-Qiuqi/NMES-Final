"""Stream the paper's 64x64, two-level SQG trajectory to an XESN-ready Zarr store.

Requires the public ``sqgturb`` module (``from sqgturb import SQG, irfft2``).
No trajectory is loaded in full memory. The stored field is raw theta anomaly
in kelvin, dimensioned x,y,z,time; training normalization is saved separately.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import xarray as xr
import yaml


def initial_pv(cfg: dict) -> np.ndarray:
    """Use the random two-level PV initial condition in rcgfd/sqgturbgenerator.py."""
    n = int(cfg["nx"])
    rng = np.random.RandomState(int(cfg["seed"]))
    pv = rng.normal(0.0, float(cfg["noise_scale"]), size=(2, n, n)).astype(np.float32)

    xx = np.linspace(0.0, 2.0 * np.pi, n, dtype=np.float32)
    yy = np.linspace(0.0, 2.0 * np.pi, n, dtype=np.float32)
    xx, yy = np.meshgrid(xx, yy)
    e = int(cfg["perturbation_exponent"])
    pv[1] += float(cfg["perturbation_amplitude"]) * np.sin(xx / 2.0) ** (2 * e) * np.sin(yy) ** e

    # Match the original generator's vertical-mean removal.
    pv -= np.nanmean(pv, axis=0, keepdims=True)
    return pv


def write_block(store_path: Path, theta: np.ndarray, times: np.ndarray,
                x: np.ndarray, y: np.ndarray, z: np.ndarray,
                first: bool, chunk_steps: int, dt_seconds: int) -> None:
    """Write a (time,z,y,x) memory block as named x,y,z,time data."""
    # Input buffer (time,z,y,x) -> XESN-friendly dimensions (x,y,z,time).
    block = theta.transpose(3, 2, 1, 0)
    ds = xr.Dataset(
        {"theta": (("x", "y", "z", "time"), block)},
        coords={"x": x, "y": y, "z": z, "time": times},
        attrs={
            "description": "SQG potential temperature anomaly, raw physical units",
            "units": "K",
            "model_dt_seconds": int(dt_seconds),
        },
    )
    ds["theta"].attrs.update({"units": "K", "description": "temperature anomaly"})
    ds.to_zarr(
        str(store_path), mode="w" if first else "a",
        append_dim=None if first else "time",
        # Match XESN's 8x8 spatial tiles to avoid reading whole-domain chunks per tile.
        encoding={"theta": {"chunks": (8, 8, 2, int(chunk_steps))}},
    )


def generate(output: Path, config_path: Path, chunk_steps: int | None = None) -> None:
    try:
        from sqgturb import SQG, irfft2
    except ImportError as exc:
        raise ImportError(
            "Install the upstream jswhit/sqgturb model and make its `sqgturb` module importable."
        ) from exc

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))["sqg"]
    if int(cfg["nx"]) != 64 or int(cfg["nz"]) != 2:
        raise ValueError("This reproduction config is for the paper's Nx=64, Nz=2 case.")

    n = int(cfg["nx"])
    dt = int(cfg["dt_seconds"])
    steps_per_year = int(cfg["days_per_year"] * 86400 / dt)
    spinup_steps = int(cfg["spinup_days"] * 86400 / dt)
    total_steps = int(cfg["years"] * steps_per_year)
    train_steps = int(15 * steps_per_year)
    chunk_steps = int(chunk_steps or cfg["output_chunk_steps"])
    if chunk_steps < 1:
        raise ValueError("output chunk length must be positive")

    # The upstream SQG public example uses this constructor and model.advance API.
    model = SQG(
        initial_pv(cfg),
        nsq=float(cfg["nsq"]), f=float(cfg["f"]), U=float(cfg["jet_speed"]),
        H=float(cfg["depth_m"]), r=0.0, tdiab=float(cfg["diabatic_days"]) * 86400.0,
        dt=dt, diff_order=int(cfg["diff_order"]),
        diff_efold=float(cfg["diff_efold_seconds"]),
        threads=int(os.environ.get("OMP_NUM_THREADS", "1")),
        precision="single", tstart=0, symmetric=bool(cfg["symmetric"]),
    )

    print(f"Spin-up: {spinup_steps:,} steps ({cfg['spinup_days']} days)")
    model.advance(timesteps=spinup_steps)

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    length_m = 20.0 * math.sqrt(float(cfg["nsq"])) * float(cfg["depth_m"]) / float(cfg["f"])
    x = np.arange(n, dtype=np.float32) * (length_m / n) / 1000.0
    y = x.copy()
    z = np.array([0.0, float(cfg["depth_m"]) / 1000.0], dtype=np.float32)
    theta_scale = float(cfg["f"]) * float(cfg["theta0"]) / float(cfg["gravity"])

    running_sum = 0.0
    running_sumsq = 0.0
    running_count = 0
    saved = 0
    first = True
    buffer = np.empty((chunk_steps, 2, n, n), dtype=np.float32)

    def collect_theta() -> np.ndarray:
        # sqgturb.irfft2 maps the model's spectral PV at both vertical levels to grid space.
        return (np.asarray(irfft2(model.pvspec), dtype=np.float32) * theta_scale)

    def flush(n_buffer: int) -> None:
        nonlocal first, saved, running_sum, running_sumsq, running_count
        if n_buffer == 0:
            return
        values = buffer[:n_buffer]
        # Include snapshots at both ends of the 15-year training interval.
        train_count = max(0, min(n_buffer, train_steps + 1 - saved))
        if train_count:
            train_values = values[:train_count].astype(np.float64, copy=False)
            running_sum += float(np.nansum(train_values))
            running_sumsq += float(np.nansum(train_values * train_values))
            running_count += int(np.isfinite(train_values).sum())
        t = (saved + np.arange(n_buffer, dtype=np.int64)) * dt
        write_block(output, values, t, x, y, z, first, chunk_steps, dt)
        first = False
        saved += n_buffer

    # Save the spun-up state at t=0, then every 300 seconds for 25 model years.
    buffer[0] = collect_theta()
    filled = 1
    for step in range(total_steps):
        model.advance(timesteps=1)
        buffer[filled] = collect_theta()
        filled += 1
        if filled == chunk_steps:
            flush(filled)
            filled = 0
        if (step + 1) % steps_per_year == 0:
            print(f"Generated {(step + 1) // steps_per_year}/{cfg['years']} years")
    flush(filled)

    if saved != total_steps + 1:
        raise RuntimeError(f"Expected {total_steps + 1:,} time samples, wrote {saved:,}.")
    if running_count == 0:
        raise RuntimeError("No finite values found in the training segment.")

    mean = running_sum / running_count
    variance = max(running_sumsq / running_count - mean * mean, 0.0)
    sd = math.sqrt(variance)
    if not math.isfinite(sd) or sd <= 0:
        raise RuntimeError(f"Invalid training standard deviation: {sd}")

    metadata = {
        "field_name": "theta",
        "units": "K",
        "global_training_mean": mean,
        "global_training_sd": sd,
        "normalization": {"bias": mean, "scale": sd},
        "dt_seconds": dt,
        "raw_time_samples": saved,
        "train_time_samples_including_endpoint": train_steps + 1,
        "train_years": 15,
        "spinup_days_discarded": int(cfg["spinup_days"]),
        "normalization_scope": "all x,y,z and first 15 years, raw theta",
    }
    metadata_path = output.with_name("normalization.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote raw theta to {output}")
    print(f"Wrote training normalization (mean={mean:.8g}, sd={sd:.8g}) to {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/sqg64_300s.zarr"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--chunk-steps", type=int, default=None)
    args = parser.parse_args()
    generate(args.output, args.config, args.chunk_steps)


if __name__ == "__main__":
    main()

