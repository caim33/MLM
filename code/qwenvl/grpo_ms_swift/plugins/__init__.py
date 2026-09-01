"""Reward plugins for Motion-r1 GRPO on ms-swift."""

from .group_bonus_vm_v import (  # noqa: F401
    GroupBonusConfig,
    RolloutScoreRecord,
    apply_vm_v_bonus_to_rewards,
    compute_vm_v_group_bonus,
)
from .rewards_semantic_format import (  # noqa: F401
    ClipTextSemanticBackend,
    format_reward_plugin,
    option_accuracy_reward_plugin,
    semantic_reward_plugin,
    set_default_semantic_backend,
)
try:
    from .swift_external_rewards import (  # noqa: F401
        MotionFormatORM,
        MotionOptionAccuracyORM,
        MotionSemanticORM,
        MotionVMVGroupBonusORM,
    )
except ModuleNotFoundError as exc:  # CPU/core environments need not install ms-swift.
    if exc.name != "swift" and not str(exc.name).startswith("swift."):
        raise
