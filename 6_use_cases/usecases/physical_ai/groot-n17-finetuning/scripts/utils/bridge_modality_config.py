"""
Modality configuration for BridgeData V2 embodiment.
Used by GR00T N1.7 fine-tuning to define how data columns map to model inputs.

This config tells GR00T:
- Which camera views to use (front only)
- State/action dimensions and how to slice them
- Action representation (absolute, non-EEF, 7-DoF)
- Language annotation key
- Temporal sampling (action horizon = 16 steps)

The modality_keys here must match the keys in meta/modality.json of the dataset.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

bridge_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["arm"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["arm"],
        action_configs=[
            # 7-DoF arm: x, y, z, roll, pitch, yaw, gripper
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(bridge_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
