import enum

from dataclasses import asdict, dataclass, field
from typing import Optional, Union

from peft.config import PeftConfigMixin


class PeftType(str, enum.Enum):
    PROMPT_TUNING = "PROMPT_TUNING"
    P_TUNING = "P_TUNING"
    PREFIX_TUNING = "PREFIX_TUNING"
    LORA = "LORA"
    ADALORA = "ADALORA"


class TaskType(str, enum.Enum):
    SEQ_CLS = "SEQ_CLS"
    SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
    CAUSAL_LM = "CAUSAL_LM"
    TOKEN_CLS = "TOKEN_CLS"


@dataclass
class PeftConfig(PeftConfigMixin):
    """
    This is the base configuration class to store the configuration of a [`PeftModel`].

    Args:
        peft_type (Union[[`~peft.utils.config.PeftType`], `str`]): The type of Peft method to use.
        task_type (Union[[`~peft.utils.config.TaskType`], `str`]): The type of task to perform.
        inference_mode (`bool`, defaults to `False`): Whether to use the Peft model in inference mode.
    """

    base_model_name_or_path: str = field(default=None, metadata={"help": "The name of the base model to use."})
    peft_type: Union[str, PeftType] = field(default=None, metadata={"help": "Peft type"})
    task_type: Union[str, TaskType] = field(default=None, metadata={"help": "Task type"})
    inference_mode: bool = field(default=False, metadata={"help": "Whether to use inference mode"})


@dataclass
class PromptLearningConfig(PeftConfig):
    """
    This is the base configuration class to store the configuration of [`PrefixTuning`], [`PromptEncoder`], or
    [`PromptTuning`].

    Args:
        num_virtual_tokens (`int`): The number of virtual tokens to use.
        token_dim (`int`): The hidden embedding dimension of the base transformer model.
        num_transformer_submodules (`int`): The number of transformer submodules in the base transformer model.
        num_attention_heads (`int`): The number of attention heads in the base transformer model.
        num_layers (`int`): The number of layers in the base transformer model.
    """

    num_virtual_tokens: int = field(default=None, metadata={"help": "Number of virtual tokens"})
    token_dim: int = field(
        default=None, metadata={"help": "The hidden embedding dimension of the base transformer model"}
    )
    num_transformer_submodules: Optional[int] = field(
        default=None, metadata={"help": "Number of transformer submodules"}
    )
    num_attention_heads: Optional[int] = field(default=None, metadata={"help": "Number of attention heads"})
    num_layers: Optional[int] = field(default=None, metadata={"help": "Number of transformer layers"})
