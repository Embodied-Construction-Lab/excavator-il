"""Small CPU-only ACT compatibility check for the excavator feature contract."""

from dataclasses import dataclass

import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


@dataclass(frozen=True)
class ActSmokeResult:
    loss: float
    predicted_chunk_shape: tuple[int, ...]


def run_act_smoke_train_step(
    *,
    image_shape: tuple[int, int, int],
    state_dim: int,
    action_dim: int,
    chunk_size: int,
) -> ActSmokeResult:
    """Run one optimizer step and one inference pass using the production feature shapes."""
    torch.manual_seed(0)
    config = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=image_shape),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
        },
        device="cpu",
        push_to_hub=False,
        chunk_size=chunk_size,
        n_action_steps=1,
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        dim_model=64,
        n_heads=4,
        dim_feedforward=128,
        n_encoder_layers=1,
        n_decoder_layers=1,
        latent_dim=16,
        n_vae_encoder_layers=1,
    )
    policy = ACTPolicy(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

    training_batch = {
        "observation.state": torch.randn(2, state_dim),
        "observation.images.front": torch.rand(2, *image_shape),
        "action": torch.empty(2, chunk_size, action_dim).uniform_(-1.0, 1.0),
        "action_is_pad": torch.zeros(2, chunk_size, dtype=torch.bool),
    }
    policy.train()
    loss, _ = policy(training_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    inference_batch = {
        "observation.state": torch.randn(1, state_dim),
        "observation.images.front": torch.rand(1, *image_shape),
    }
    with torch.no_grad():
        predicted_chunk = policy.predict_action_chunk(inference_batch)

    return ActSmokeResult(
        loss=float(loss.detach()),
        predicted_chunk_shape=tuple(predicted_chunk.shape),
    )
