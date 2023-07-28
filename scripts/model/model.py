# pylint: disable=E1102

"""
Custom Trainer model that support weights for classification loss function.
"""

import torch
from transformers import Trainer


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.device = device

    def compute_loss(self, model, inputs, return_outputs: bool = False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(self.class_weights)
        ).to(self.device)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
