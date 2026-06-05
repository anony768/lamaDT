

from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMBackbone:

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
                                               
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if device is None else None,
        )

        if device is not None:
            self.model.to(device)

        for p in self.model.parameters():
            p.requires_grad = False

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
        **kwargs: Dict[str, Any],
    ):
        
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

