

from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import AutoTokenizer

class TaskIDEmbedding(nn.Module):

    def __init__(self, num_tasks: int, hidden_size: int, num_tokens: int = 4):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_tokens = num_tokens
        self.embedding = nn.Embedding(num_tasks, hidden_size * num_tokens)
        self.hidden_size = hidden_size

    def forward(self, task_ids: torch.Tensor) -> torch.Tensor:
        
        emb = self.embedding(task_ids)                                 
        return emb.view(-1, self.num_tokens, self.hidden_size)

class NumericEmbedding(nn.Module):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
        )
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, hidden_size),
            nn.Tanh(),
        )
        self.return_mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
        )

    def embed_state(self, s: torch.Tensor) -> torch.Tensor:
        return self.state_mlp(s)

    def embed_action(self, a: torch.Tensor) -> torch.Tensor:
        return self.action_mlp(a)

    def embed_return(self, rtg: torch.Tensor) -> torch.Tensor:
                       
        return self.return_mlp(rtg)

class NumericHeads(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        state_dim: int,
        action_dim: int,
    ) -> None:
        super().__init__()
        self.state_head = nn.Linear(hidden_size, state_dim)
        self.action_head = nn.Linear(hidden_size, action_dim)
        self.reward_head = nn.Linear(hidden_size, 1)

    def predict_next_state(self, h: torch.Tensor) -> torch.Tensor:
        return self.state_head(h)

    def predict_action(self, h: torch.Tensor) -> torch.Tensor:
        return self.action_head(h)

    def predict_reward(self, h: torch.Tensor) -> torch.Tensor:
        return self.reward_head(h)

class TrajectoryTokenizer:

    def __init__(
        self,
        llm_tokenizer_name: str,
        state_dim: int,
        action_dim: int,
        hidden_size: int,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.numeric_embedding = NumericEmbedding(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
        )
        self.numeric_heads = NumericHeads(
            hidden_size=hidden_size,
            state_dim=state_dim,
            action_dim=action_dim,
        )

    def encode_prefix(
        self,
        task_description: str,
        provenance_token: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        parts: List[str] = [task_description]
        if provenance_token is not None:
            parts.append(f"[source={provenance_token}]")
        text = " ".join(parts)
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return enc["input_ids"], enc["attention_mask"]

    def encode_trajectory_numeric(
        self,
        rtg: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        e_r = self.numeric_embedding.embed_return(rtg)
        e_s = self.numeric_embedding.embed_state(states)
        e_a = self.numeric_embedding.embed_action(actions)
        return e_r, e_s, e_a

    def build_dt_sequence(
        self,
        word_embed_fn: nn.Module,
        text_ids: torch.Tensor,
        rtgs: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        device: torch.device,
        text_attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        
        B, T = states.shape[0], states.shape[1]

        text_embeds = word_embed_fn(text_ids.to(device)).float()
        L_text = text_embeds.shape[1]

        if text_embeds.shape[0] == 1 and B > 1:
            text_embeds = text_embeds.expand(B, -1, -1)

        if text_attn_mask is not None:
            text_attn = text_attn_mask.to(device)
        else:
            text_attn = torch.ones(
                text_embeds.shape[0], L_text, device=device, dtype=torch.long,
            )
            if text_attn.shape[0] == 1 and B > 1:
                text_attn = text_attn.expand(B, -1)

        flat_r = rtgs.reshape(B * T, 1).to(device)
        flat_s = states.reshape(B * T, -1).to(device)
        flat_a = actions.reshape(B * T, -1).to(device)

        e_r = self.numeric_embedding.embed_return(flat_r).reshape(B, T, -1)
        e_s = self.numeric_embedding.embed_state(flat_s).reshape(B, T, -1)
        e_a = self.numeric_embedding.embed_action(flat_a).reshape(B, T, -1)

        numeric = torch.stack([e_r, e_s, e_a], dim=2).reshape(B, T * 3, -1)

        combined = torch.cat([text_embeds, numeric.float()], dim=1)
        numeric_attn = torch.ones(B, T * 3, device=device, dtype=torch.long)
        attn = torch.cat([text_attn, numeric_attn], dim=1)

        return combined, attn, L_text

    def build_dt_sequence_taskid(
        self,
        task_id_embeds: torch.Tensor,
        rtgs: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        
        B, T = states.shape[0], states.shape[1]
        L_prefix = task_id_embeds.shape[1]

        text_embeds = task_id_embeds.float().to(device)
        text_attn = torch.ones(B, L_prefix, device=device, dtype=torch.long)

        flat_r = rtgs.reshape(B * T, 1).to(device)
        flat_s = states.reshape(B * T, -1).to(device)
        flat_a = actions.reshape(B * T, -1).to(device)

        e_r = self.numeric_embedding.embed_return(flat_r).reshape(B, T, -1)
        e_s = self.numeric_embedding.embed_state(flat_s).reshape(B, T, -1)
        e_a = self.numeric_embedding.embed_action(flat_a).reshape(B, T, -1)

        numeric = torch.stack([e_r, e_s, e_a], dim=2).reshape(B, T * 3, -1)
        combined = torch.cat([text_embeds, numeric.float()], dim=1)
        numeric_attn = torch.ones(B, T * 3, device=device, dtype=torch.long)
        attn = torch.cat([text_attn, numeric_attn], dim=1)

        return combined, attn, L_prefix
