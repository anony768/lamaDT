

from typing import Optional, Tuple

import torch
from torch import nn

class LoRAAdapter(nn.Module):

    def __init__(self, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
                                            
        self.lora_layers: nn.ModuleDict = nn.ModuleDict()
                                                            
        self.enabled: bool = True

    def _make_lora_for_linear(self, linear: nn.Linear) -> Tuple[nn.Linear, nn.Linear]:
        
        in_dim = linear.in_features
        out_dim = linear.out_features
        lora_A = nn.Linear(in_dim, self.rank, bias=False)
        lora_B = nn.Linear(self.rank, out_dim, bias=False)
                                 
        nn.init.zeros_(lora_B.weight)
        return lora_A, lora_B

    def attach_to_llama(
        self,
        model: nn.Module,
        target_suffixes: Tuple[str, ...] = ("o_proj", "down_proj"),
    ) -> None:

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not any(name.endswith(suf) for suf in target_suffixes):
                continue

            lora_A, lora_B = self._make_lora_for_linear(module)
            key = name.replace(".", "_")
            device = module.weight.device
                                                                                           
            self.lora_layers[key] = nn.Sequential(lora_A, lora_B).to(device=device)

            def _make_hook(adapter_ref, layer_key: str):
                def hook(mod, inp, out):
                    if not adapter_ref.enabled:
                        return out
                    x = inp[0].float()                                          
                    seq = adapter_ref.lora_layers[layer_key]
                    delta = seq(x) * (adapter_ref.alpha / adapter_ref.rank)
                    delta = adapter_ref.dropout(delta).to(out.dtype)                           
                    return out + delta

                return hook

            module.register_forward_hook(_make_hook(self, key))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        
        return hidden_states

class ObjectiveAdapters(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.objective_adapters: nn.ModuleDict = nn.ModuleDict()
        self.shared_adapter: Optional[LoRAAdapter] = None

    def register_objective(self, name: str, adapter: LoRAAdapter) -> None:
        self.objective_adapters[name] = adapter

    def set_shared_adapter(self, adapter: LoRAAdapter) -> None:
        self.shared_adapter = adapter

    def get_active_adapters(self, objective_name: str):
        
        adapter = self.objective_adapters[objective_name] if objective_name in self.objective_adapters else None
        return adapter, self.shared_adapter

    def activate(self, objective_name: str) -> None:
        
        for name, adapter in self.objective_adapters.items():
            adapter.enabled = (name == objective_name)
        if self.shared_adapter is not None:
            self.shared_adapter.enabled = True

    def enable_all(self) -> None:
        
        for adapter in self.objective_adapters.values():
            adapter.enabled = True
        if self.shared_adapter is not None:
            self.shared_adapter.enabled = True

