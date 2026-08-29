"""Small reproducible benchmark for the ACT dataset input pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any, Callable


@dataclass(frozen=True)
class TrainingInputBenchmark:
    dataset_root: str
    repo_id: str
    total_frames: int
    total_episodes: int
    camera_dtypes: tuple[str, ...]
    batch_size: int
    num_workers: int
    warmup_batches: int
    measured_batches: int
    measured_samples: int
    elapsed_s: float
    batches_per_s: float
    samples_per_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dataset_factory(repo_id: str, *, root: Path) -> Any:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(repo_id=repo_id, root=root)


def _loader_factory(**kwargs: Any) -> Any:
    from torch.utils.data import DataLoader

    return DataLoader(**kwargs)


def _integer(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _next_batch(iterator: Any) -> Any:
    try:
        return next(iterator)
    except StopIteration as exc:
        raise ValueError("dataset ended before the benchmark completed") from exc


def _observed_batch_size(batch: Any, fallback: int) -> int:
    if isinstance(batch, dict):
        for value in batch.values():
            shape = getattr(value, "shape", None)
            if shape is not None and len(shape) > 0:
                return int(shape[0])
    return fallback


def benchmark_training_input(
    dataset_root: str | Path,
    repo_id: str,
    *,
    batch_size: int,
    num_workers: int,
    warmup_batches: int = 5,
    measured_batches: int = 20,
    dataset_factory: Callable[..., Any] = _dataset_factory,
    loader_factory: Callable[..., Any] = _loader_factory,
    clock: Callable[[], float] = time.perf_counter,
) -> TrainingInputBenchmark:
    """Measure decoded sample throughput without constructing an ACT policy."""

    batch_size = _integer("batch_size", batch_size, minimum=1)
    num_workers = _integer("num_workers", num_workers, minimum=0)
    warmup_batches = _integer("warmup_batches", warmup_batches, minimum=0)
    measured_batches = _integer(
        "measured_batches", measured_batches, minimum=1
    )
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError("repo_id must be a non-empty string")
    root = Path(dataset_root).resolve()
    dataset = dataset_factory(repo_id.strip(), root=root)
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    iterator = iter(loader_factory(**loader_kwargs))
    for _ in range(warmup_batches):
        _next_batch(iterator)
    started = clock()
    measured_samples = 0
    for _ in range(measured_batches):
        batch = _next_batch(iterator)
        measured_samples += _observed_batch_size(batch, batch_size)
    elapsed = clock() - started
    if elapsed <= 0.0:
        raise ValueError("benchmark clock did not advance")
    features = getattr(dataset.meta, "features", {})
    camera_dtypes = tuple(
        sorted(
            str(feature.get("dtype"))
            for name, feature in features.items()
            if name.startswith("observation.images.")
            and isinstance(feature, dict)
        )
    )
    return TrainingInputBenchmark(
        dataset_root=str(root),
        repo_id=repo_id.strip(),
        total_frames=int(dataset.meta.total_frames),
        total_episodes=int(dataset.meta.total_episodes),
        camera_dtypes=camera_dtypes,
        batch_size=batch_size,
        num_workers=num_workers,
        warmup_batches=warmup_batches,
        measured_batches=measured_batches,
        measured_samples=measured_samples,
        elapsed_s=elapsed,
        batches_per_s=measured_batches / elapsed,
        samples_per_s=measured_samples / elapsed,
    )
