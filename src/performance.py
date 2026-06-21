from __future__ import annotations

from dataclasses import dataclass, asdict
import importlib.util
import math
import os
from typing import Literal


Engine = Literal["joblib", "dask"]
TaskName = Literal["grid", "prevention", "route", "benchmark", "generic"]


@dataclass(frozen=True)
class RuntimeConfig:
    engine: Engine
    n_jobs: int
    max_cpu: int
    physical_cpu: int | None
    logical_cpu: int
    memory_gb: float | None
    estimated_rows: int
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def _memory_gb() -> float | None:
    try:
        import psutil
        return round(psutil.virtual_memory().available / (1024 ** 3), 2)
    except Exception:
        return None


def _physical_cpu() -> int | None:
    try:
        import psutil
        return psutil.cpu_count(logical=False)
    except Exception:
        return None


def _is_dask_available() -> bool:
    return importlib.util.find_spec("dask") is not None


def detect_runtime_limits(user_cap: int = 8) -> tuple[int, int | None, int, float | None]:
    logical = os.cpu_count() or 1
    physical = _physical_cpu()
    memory = _memory_gb()
    max_cpu = max(1, min(int(user_cap), logical))
    return max_cpu, physical, logical, memory


def recommend_workers(
    rows: int,
    task: TaskName = "generic",
    user_cap: int = 8,
    reserve_cores: int = 1,
) -> int:
    """Estimare conservatoare pentru numărul de workeri.

    Ideea este să nu folosim automat toate nucleele, deoarece Streamlit, browserul,
    sistemul de operare și bibliotecile geospațiale au și ele overhead.
    """
    max_cpu, physical, logical, memory = detect_runtime_limits(user_cap=user_cap)
    usable_cpu = max(1, min(max_cpu, (physical or logical) - max(0, reserve_cores)))

    rows = max(0, int(rows))
    if rows < 25_000:
        cpu_by_rows = 1
    elif rows < 100_000:
        cpu_by_rows = 2
    elif rows < 350_000:
        cpu_by_rows = min(4, usable_cpu)
    else:
        cpu_by_rows = usable_cpu

    # Pentru aplicația principală și benchmark folosim aceeași logică de estimare.
    # Limităm implicit la 8 workeri, ca să fie stabil pe laptopuri, dar fără diferențe
    # artificiale între app și benchmark.
    if task in {"prevention", "route", "grid", "generic"}:
        cpu_by_rows = min(cpu_by_rows, 8)
    elif task == "benchmark":
        cpu_by_rows = min(cpu_by_rows, min(usable_cpu, 8))

    # Limită simplă de memorie: aproximăm că fiecare worker are nevoie de spațiu
    # pentru copii/intermediare ale dataframe-urilor. Este intenționat conservator.
    if memory is not None:
        if memory < 4:
            cpu_by_rows = min(cpu_by_rows, 2)
        elif memory < 8:
            cpu_by_rows = min(cpu_by_rows, 4)

    return max(1, int(cpu_by_rows))


def recommend_engine(rows: int, task: TaskName = "generic", prefer_distributed: bool = True) -> Engine:
    """Alege între Joblib și Dask.

    Joblib este implicit pentru operații geospațiale și funcții cu shapely/geopandas,
    deoarece overhead-ul Dask nu merită mereu. Dask este recomandat doar pentru agregări
    mari de tip dataframe, de exemplu grid pe multe rânduri.
    """
    rows = int(rows)
    if (
        prefer_distributed
        and task == "grid"
        and rows >= 350_000
        and _is_dask_available()
    ):
        return "dask"
    return "joblib"


def auto_runtime_config(
    rows: int,
    task: TaskName = "generic",
    user_cap: int = 8,
    prefer_distributed: bool = True,
) -> RuntimeConfig:
    max_cpu, physical, logical, memory = detect_runtime_limits(user_cap=user_cap)
    n_jobs = recommend_workers(rows, task=task, user_cap=user_cap)
    engine = recommend_engine(rows, task=task, prefer_distributed=prefer_distributed)

    if engine == "dask":
        reason = (
            f"Dataset mare ({rows:,} rânduri) și Dask disponibil; folosesc Dask pentru agregări dataframe."
        )
    elif n_jobs == 1:
        reason = (
            f"Dataset mic sau resurse limitate; folosesc 1 worker pentru overhead minim."
        )
    else:
        reason = (
            f"Estimare conservatoare: {n_jobs} workeri din {logical} CPU logice, cu rezervă pentru sistem și interfață."
        )

    return RuntimeConfig(
        engine=engine,
        n_jobs=n_jobs,
        max_cpu=max_cpu,
        physical_cpu=physical,
        logical_cpu=logical,
        memory_gb=memory,
        estimated_rows=int(rows),
        reason=reason,
    )


def explain_runtime_config(config: RuntimeConfig) -> str:
    ram = "necunoscută" if config.memory_gb is None else f"~{config.memory_gb} GB RAM disponibili"
    physical = "necunoscut" if config.physical_cpu is None else str(config.physical_cpu)
    return (
        f"Auto: engine={config.engine}, workeri={config.n_jobs}. "
        f"CPU logic={config.logical_cpu}, CPU fizic={physical}, {ram}. "
        f"{config.reason}"
    )


def candidate_worker_list(rows: int, user_cap: int = 8) -> list[int]:
    """Listă scurtă pentru benchmark rapid pe un calculator nou."""
    recommended = recommend_workers(rows, user_cap=user_cap)
    values = {1, recommended}
    for value in [2, 4, 6, 8, os.cpu_count() or 1]:
        if 1 <= value <= user_cap:
            values.add(value)
    return sorted(values)
