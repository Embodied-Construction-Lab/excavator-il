import math

import pytest

pytest.importorskip("lerobot", reason="install excavator-il[training] for ACT tests")

from excavator_il.act_smoke import run_act_smoke_train_step


def test_act_smoke_train_step_accepts_rgb_state_and_four_axis_action():
    result = run_act_smoke_train_step(
        image_shape=(3, 32, 32),
        state_dim=11,
        action_dim=4,
        chunk_size=3,
    )

    assert math.isfinite(result.loss)
    assert result.loss > 0.0
    assert result.predicted_chunk_shape == (1, 3, 4)
