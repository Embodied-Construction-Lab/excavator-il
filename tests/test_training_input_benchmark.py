from pathlib import Path
from types import SimpleNamespace

import pytest

from excavator_il.training_input_benchmark import benchmark_training_input


class _Clock:
    def __init__(self, values):
        self._values = iter(values)

    def __call__(self):
        return next(self._values)


def test_reports_batches_and_samples_per_second(tmp_path: Path):
    dataset = SimpleNamespace(
        meta=SimpleNamespace(
            total_frames=100,
            total_episodes=5,
            features={"observation.images.front": {"dtype": "video"}},
        )
    )
    calls = []

    def dataset_factory(repo_id, *, root):
        calls.append((repo_id, root))
        return dataset

    result = benchmark_training_input(
        tmp_path,
        "local/train",
        batch_size=4,
        num_workers=2,
        warmup_batches=1,
        measured_batches=2,
        dataset_factory=dataset_factory,
        loader_factory=lambda **kwargs: iter([object(), object(), object()]),
        clock=_Clock([10.0, 12.0]),
    )

    assert calls == [("local/train", tmp_path.resolve())]
    assert result.measured_batches == 2
    assert result.measured_samples == 8
    assert result.elapsed_s == 2.0
    assert result.batches_per_s == 1.0
    assert result.samples_per_s == 4.0
    assert result.camera_dtypes == ("video",)


@pytest.mark.parametrize(
    ("field", "value"),
    (("batch_size", 0), ("num_workers", -1), ("measured_batches", True)),
)
def test_rejects_invalid_numeric_inputs(tmp_path: Path, field: str, value: object):
    kwargs = {
        "batch_size": 2,
        "num_workers": 0,
        "warmup_batches": 0,
        "measured_batches": 1,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=field):
        benchmark_training_input(
            tmp_path,
            "local/train",
            dataset_factory=lambda *_args, **_kwargs: None,
            **kwargs,
        )


def test_rejects_empty_repo_and_non_advancing_clock(tmp_path: Path):
    with pytest.raises(ValueError, match="repo_id"):
        benchmark_training_input(
            tmp_path,
            " ",
            batch_size=1,
            num_workers=0,
        )

    dataset = SimpleNamespace(
        meta=SimpleNamespace(total_frames=1, total_episodes=1, features={})
    )
    with pytest.raises(ValueError, match="clock did not advance"):
        benchmark_training_input(
            tmp_path,
            "local/train",
            batch_size=1,
            num_workers=0,
            warmup_batches=0,
            measured_batches=1,
            dataset_factory=lambda *_args, **_kwargs: dataset,
            loader_factory=lambda **_kwargs: iter([object()]),
            clock=_Clock([4.0, 4.0]),
        )


def test_rejects_dataset_that_ends_during_measurement(tmp_path: Path):
    dataset = SimpleNamespace(
        meta=SimpleNamespace(total_frames=1, total_episodes=1, features={})
    )

    with pytest.raises(ValueError, match="dataset ended"):
        benchmark_training_input(
            tmp_path,
            "local/train",
            batch_size=1,
            num_workers=0,
            warmup_batches=0,
            measured_batches=2,
            dataset_factory=lambda *_args, **_kwargs: dataset,
            loader_factory=lambda **_kwargs: iter([object()]),
        )
