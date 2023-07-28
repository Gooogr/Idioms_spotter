"""
Data class arguments for training script
"""

from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )


@dataclass
class DataTrainArguments:
    dataset_name: str = field(
        default="Gooogr/pie_idioms",
        metadata={"help": "Dataset identifier from huggingface.co/datasets"},
    )


@dataclass
class PeftArguments:
    use_lora: bool = field(
        default=True,
        metadata={"help": "Boolean flag to use LoRA PEFT approach"},
    )
    r: int = field(
        default=8,
        metadata={"help": "The dimension of the low-rank attention matrices"},
    )
    lora_alpha: int = field(
        default=8,
        metadata={"help": "Scaling factor for the weight matrices"},
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout probability for Lora layers"},
    )
    bias: str = field(
        default="none",
        metadata={
            "help": """Bias type for Lora. Can be ‘none’, ‘all’ or ‘lora_only’.
                  If ‘all’ or ‘lora_only’, the corresponding biases will be updated during training.
                  Be aware that this means that, even when disabling the adapters, the model will not produce
                  the same output as the base model would have without adaptation."""
        },
    )
