from __future__ import annotations

import time
from statistics import mean, stdev
from typing import Callable, Any

import pandas as pd

from .performance import candidate_worker_list


def timed_run(name: str, func: Callable[[], Any], repeats: int = 1) -> dict:
    """Rulează o funcție și întoarce statistici simple de timp."""
    times: list[float] = []
    rows_out = None
    error = ""

    for _ in range(max(1, int(repeats))):
        start = time.perf_counter()
        try:
            result = func()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            if hasattr(result, "__len__"):
                rows_out = len(result)
        except Exception as exc:  # benchmark-ul nu trebuie să oprească aplicația
            error = repr(exc)
            break

    return {
        "task": name,
        "status": "ok" if not error else "eroare",
        "repeats": len(times),
        "mean_s": round(mean(times), 4) if times else None,
        "std_s": round(stdev(times), 4) if len(times) > 1 else 0.0 if times else None,
        "min_s": round(min(times), 4) if times else None,
        "max_s": round(max(times), 4) if times else None,
        "rows_out": rows_out,
        "error": error,
    }


def add_speedup_columns(results: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    out["speedup_vs_1_worker"] = None
    out["efficiency_pct"] = None
    ok = out[out["status"] == "ok"].copy()
    if ok.empty or "workers" not in ok.columns:
        return out

    for task, group in ok.groupby("task"):
        group = group.sort_values("workers")
        baseline_row = group[group["workers"] == 1]
        if baseline_row.empty:
            baseline = group.iloc[0]
        else:
            baseline = baseline_row.iloc[0]
        baseline_time = float(baseline["mean_s"])
        baseline_workers = int(baseline["workers"])
        mask = (out["task"] == task) & out["mean_s"].notna()
        out.loc[mask, "speedup_vs_1_worker"] = out.loc[mask].apply(
            lambda row: round(baseline_time / float(row["mean_s"]), 3)
            if float(row["mean_s"]) > 0 else None,
            axis=1,
        )
        out.loc[mask, "efficiency_pct"] = out.loc[mask].apply(
            lambda row: round(
                float(row["speedup_vs_1_worker"]) / max(int(row["workers"]) / baseline_workers, 1) * 100,
                2,
            ) if row["speedup_vs_1_worker"] is not None else None,
            axis=1,
        )
    return out


def suggested_benchmark_workers(rows: int, user_cap: int = 8) -> list[int]:
    return candidate_worker_list(rows, user_cap=user_cap)
