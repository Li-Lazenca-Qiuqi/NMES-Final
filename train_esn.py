"""Optimize and train one manuscript-style local ESN configuration."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from mean_ridge import install_mean_covariance_solver


def load_ke_diagnostic():
    """Load rcgfd/xsqgturb.py without importing rcgfd.__init__ (which needs ddc)."""
    repo = Path(__file__).resolve().parents[1]
    source = repo / "rcgfd" / "xsqgturb.py"
    spec = importlib.util.spec_from_file_location("reproduce_xsqgturb", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load KE diagnostic module at {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.XSQGTurb()


def paper_nrmse(prediction: xr.DataArray, truth: xr.DataArray) -> float:
    """Global-state NRMSE; inputs are normalized by the scalar training SD."""
    value = np.sqrt(((prediction - truth) ** 2).mean()).compute()
    return float(value.item())


def paper_ke_nrmse(prediction: xr.DataArray, truth: xr.DataArray,
                   mean: float, sd: float, diagnostic) -> float:
    """KE_NRMSE from main.tex, calculated in physical theta units."""
    pred_phys = (prediction * sd + mean).compute()
    truth_phys = (truth * sd + mean).compute()        #DS说先物化成 NumPy，避免 dask FFT 的分块限制
    pred_ke = diagnostic.calc_kespec1d(pred_phys)
    truth_ke = diagnostic.calc_kespec1d(truth_phys)
    time_dim = "ftime" if "ftime" in truth_ke.dims else "time"
    scale = truth_ke.std(time_dim)
    norm_error = (pred_ke - truth_ke) / scale
    norm_error = norm_error.where(scale > 0)
    result = np.sqrt((norm_error ** 2).mean(dim=("k1d", time_dim), skipna=True)).compute()
    return float(result.item())


class SQGMacroObjective:
    """Callable objective accepted by xesn.optim.optimize."""

    def __init__(self, config, train_data, macro_data, gamma, mean, sd, diagnostic):
        self.config = config
        self.train_data = train_data
        self.macro_data = macro_data
        self.gamma = float(gamma)
        self.mean = float(mean)
        self.sd = float(sd)
        self.diagnostic = diagnostic

    def __call__(self, parameter_sets, is_transformed=True):
        from xesn.lazyesn import LazyESN
        from xesn.optim import inverse_transform
        from xesn.utils import update_esn_kwargs

        parameter_sets = np.atleast_2d(parameter_sets)
        parameter_names = tuple(self.config["macro_training"]["parameters"].keys())
        values = np.empty(parameter_sets.shape[0], dtype=np.float64)
        base = self.config["lazyesn"]
        training_options = self.config["training"]
        forecast = self.config["macro_training"]["forecast"]

        for i, values_i in enumerate(parameter_sets):
            named = dict(zip(parameter_names, values_i))
            if is_transformed:
                named = inverse_transform(
                    named, self.config["macro_training"].get("transformations", {})
                )
            model = LazyESN(**update_esn_kwargs(named, base))
            model.build()
            model.train(self.train_data, **training_options)

            sample_costs = []
            for sample in self.macro_data:
                result = model.test(
                    sample,
                    n_steps=int(forecast["n_steps"]),
                    n_spinup=int(forecast["n_spinup"]),
                )
                cost = paper_nrmse(result["prediction"], result["truth"])
                if self.gamma:
                    cost += self.gamma * paper_ke_nrmse(
                        result["prediction"], result["truth"],
                        self.mean, self.sd, self.diagnostic,
                    )
                sample_costs.append(cost)
            values[i] = float(np.mean(sample_costs))
            print(f"Macro sample {i + 1}/{len(parameter_sets)}: objective={values[i]:.8g}", flush=True)

        return values.reshape(-1, 1)


def make_data_config(store: str, normalizer: dict, nsub: int, cfg: dict) -> dict:
    sqg = cfg["sqg"]
    train_end = int(cfg["training"]["train_years"] * sqg["days_per_year"] * 86400 / sqg["dt_seconds"])
    validation_end = int((cfg["training"]["train_years"] + cfg["training"]["validation_years"]) * sqg["days_per_year"] * 86400 / sqg["dt_seconds"])
    total_end = int(sqg["years"] * sqg["days_per_year"] * 86400 / sqg["dt_seconds"])
    return {
        "field_name": "theta",
        "zstore_path": store,
        "dimensions": ["x", "y", "z", "time"],
        "subsampling": {
            "time": {
                "training": [0, train_end + 1, nsub],
                "macro_training": [train_end, validation_end + 1, nsub],
                "testing": [validation_end, total_end + 1, nsub],
            }
        },
        "normalization": {
            "bias": float(normalizer["global_training_mean"]),
            "scale": float(normalizer["global_training_sd"]),
        },
    }


def load_paper_macro_indices(nmacro: int, dt_seconds: int) -> list[int]:
    """Read the authors' fixed validation forecast indices for matched Nsub cases."""
    repo = Path(__file__).resolve().parents[1]
    index_path = repo / "scripts" / "sqg-064n" / "sample-indices.yaml"
    table = yaml.safe_load(index_path.read_text(encoding="utf-8"))
    if nmacro not in table:
        raise KeyError(f"No fixed sample-index set for n_macro={nmacro} in {index_path}")
    # The checked-in YAML represents each timestep entry as a one-item list
    # (``[{300: [...]}, {600: [...]}]``), rather than one mapping. Accept both
    # that source format and a normalized mapping for clarity.
    indices_by_dt = table[nmacro]
    if isinstance(indices_by_dt, list):
        indices_by_dt = {
            int(dt): indices
            for item in indices_by_dt
            for dt, indices in item.items()
        }
    else:
        indices_by_dt = {int(dt): indices for dt, indices in indices_by_dt.items()}
    if dt_seconds not in indices_by_dt:
        raise KeyError(f"No fixed validation indices for dt={dt_seconds} s in {index_path}")
    return [int(index) for index in indices_by_dt[dt_seconds]]


def optimize_and_train(data_store: Path, output: Path, config_path: Path,
                       nsub: int, gamma: float, reservoir: int,
                       n_workers: int, threads_per_worker: int,
                       localport: int | None = None) -> None:
    from dask.distributed import Client, LocalCluster
    import xesn
    from xesn.lazyesn import LazyESN
    from xesn.optim import optimize
    from xesn.utils import get_samples, update_esn_kwargs

    if xesn._use_cupy:
        raise RuntimeError(
            "XESN's SMT Bayesian optimizer is CPU-only. Run this macro-calibration in an environment "
            "where XESN does not auto-enable CuPy, or use a CPU node."
        )

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    metadata_path = data_store.with_name("normalization.json")
    normalizer = json.loads(metadata_path.read_text(encoding="utf-8"))
    mean = float(normalizer["global_training_mean"])
    sd = float(normalizer["global_training_sd"])
    if not math.isfinite(sd) or sd <= 0:
        raise ValueError(f"Invalid normalization SD in {metadata_path}: {sd}")

    train = cfg["training"]
    sqg = cfg["sqg"]
    spinup = int(train["n_spinup_days"] * 86400 / (sqg["dt_seconds"] * nsub))
    n_steps = int(train["forecast_hours"] * 3600 / (sqg["dt_seconds"] * nsub))
    if spinup <= 0 or n_steps <= 0:
        raise ValueError("Nsub gives zero spin-up or forecast steps.")

    xdata = make_data_config(str(data_store), normalizer, nsub, cfg)
    nmacro = int(train["n_macro"])
    esn_config = {
        "esn_chunks": deepcopy(train["esn_chunks"]),
        "overlap": deepcopy(train["overlap"]),
        "boundary": train["boundary"],
        "n_reservoir": int(reservoir),
        "leak_rate": 0.5,
        "tikhonov_parameter": 1.0e-6,
        "persist": True,
        "input_kwargs": {
            "factor": 0.5,
            "distribution": "uniform",
            "normalization": "svd",
            "is_sparse": False,
            "random_seed": int(train["matrix_seed"]),
        },
        "adjacency_kwargs": {
            "factor": 0.5,
            "distribution": "uniform",
            "normalization": "svd",
            "is_sparse": True,
            "connectedness": int(train["connectedness"]),
            "format": "csr",
            "random_seed": int(train["matrix_seed"]) + 1,
        },
        "bias_kwargs": {
            "factor": 0.5,
            "distribution": "uniform",
            "random_seed": int(train["matrix_seed"]) + 2,
        },
    }

    # Use an explicit local distributed cluster. Adjust this when attaching to an HPC scheduler.
    cluster_kwargs = dict( n_workers=n_workers, 
            threads_per_worker=threads_per_worker,
            )

    if localport is not None:
        cluster_kwargs["dashboard_address"] = f":{int(localport)}"

    with LocalCluster(**cluster_kwargs) as cluster:
        with Client(cluster) as client:
            print(f"Dask dashboard available at: {client.dashboard_link}", flush=True)

            from xesn.xdata import XData
            data = XData(**xdata)
            train_data = data.setup(mode="training")
            macro_pool = data.setup(mode="macro_training")
            sample_indices = load_paper_macro_indices(
                nmacro, int(sqg["dt_seconds"] * nsub)
            )
            macro_data, sample_indices = get_samples(
                xda=macro_pool,
                n_samples=nmacro,
                n_steps=n_steps,
                n_spinup=spinup,
                sample_indices=sample_indices,
            )

            macro_config = {
                "lazyesn": esn_config,
                "training": {"n_spinup": 0, "batch_size": int(train["batch_size"])},
                "macro_training": {
                    "parameters": deepcopy(train["bounds"]),
                    "transformations": {"tikhonov_parameter": "log10"},
                    "forecast": {"n_samples": nmacro, "n_steps": n_steps, "n_spinup": spinup},
                },
            }
            diagnostic = load_ke_diagnostic() if gamma else None
            objective = SQGMacroObjective(
                macro_config, train_data, macro_data, gamma, mean, sd, diagnostic
            )

            # ... 在 install_mean_covariance_solver() 之前
            original_use_cupy = getattr(xesn, "_use_cupy", False)   # 保存原始状态
            try:

                # 临时禁用 CuPy，让贝叶斯优化在 CPU 上运行
                xesn._use_cupy = False

                install_mean_covariance_solver()
                optimized = optimize(
                    objective,
                    n_doe=int(train["n_doe"]),
                    n_iter=int(train["n_iter"]),
                    n_parallel=int(train["n_parallel"]),
                    seed=int(train["optimizer_seed"]),	#跑不通，说去掉了用默认方法优化
            )
            finally:

                # 恢复 CuPy 状态，让后续的最终模型训练使用 GPU
                xesn._use_cupy = original_use_cupy
                # ... 之后创建 final_model 并训练


            final_kwargs = update_esn_kwargs(optimized, esn_config)
            final_model = LazyESN(**final_kwargs)
            final_model.build()
            final_model.train(train_data, n_spinup=0, batch_size=int(train["batch_size"]))

            output.mkdir(parents=True, exist_ok=True)
            model_path = output / "model.zarr"
            if model_path.exists():
                raise FileExistsError(f"Refusing to overwrite model store: {model_path}")
            model_ds = final_model.to_xds()
            model_ds.attrs["nsub"] = int(nsub)
            model_ds.attrs["gamma_ke_nrmse"] = float(gamma)
            model_ds.attrs["training_normalization_mean"] = mean
            model_ds.attrs["training_normalization_sd"] = sd
            model_ds.to_zarr(str(model_path), mode="w")

    run_metadata = {
        "model_store": str(output / "model.zarr"),
        "data_store": str(data_store),
        "field_name": "theta",
        "nsub": int(nsub),
        "dt_seconds": int(sqg["dt_seconds"] * nsub),
        "gamma_ke_nrmse": float(gamma),
        "reservoir_dimension_per_tile": int(reservoir),
        "normalization": {"bias": mean, "scale": sd},
        "n_spinup": spinup,
        "n_steps_12h": n_steps,
        "macro_sample_indices": sample_indices,
        "optimized_parameters": {k: float(v) for k, v in optimized.items()},
        "readout_objective": "mean covariance + Tikhonov (paper convention)",
        "macro_objective": "global-state NRMSE + gamma * KE_NRMSE",
        "xesn_version_expected": "0.2.x source in ../timothyas-xesn-02ef51b",
    }
    (output / "run.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    print(f"Saved XESN model to {output / 'model.zarr'}")
    print(f"Saved run metadata to {output / 'run.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/sqg64_300s.zarr"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nsub", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--reservoir", type=int, default=6000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--localport", type=int, default=None, 
            help="Port on localhost for the Dask dashboard, default 8787.",
            )

    args = parser.parse_args()
    if args.nsub < 1:
        parser.error("--nsub must be a positive integer")
    if args.reservoir < 1:
        parser.error("--reservoir must be a positive integer")
    optimize_and_train(
        args.data, args.output, args.config, args.nsub, args.gamma,
        args.reservoir, args.workers, args.threads_per_worker,
        localport=args.localport,
    )


if __name__ == "__main__":
    main()

