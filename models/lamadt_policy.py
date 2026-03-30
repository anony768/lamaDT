
from typing import Any, Dict

import numpy as np
import torch

class LaMADTPolicy:
    def __init__(self, llm, policy_lora, tokenizer):
        self.llm = llm
        self.policy_lora = policy_lora
        self.tokenizer = tokenizer

    @torch.no_grad()
    def act(
        self,
        trajectory_prefix: Dict[str, np.ndarray],
        task_description: str,
        provenance_token: str,
    ) -> np.ndarray:
        device = next(self.llm.model.parameters()).device
        model_dtype = next(self.llm.model.parameters()).dtype

        if self.policy_lora is not None:
            self.policy_lora.enabled = True
            self.policy_lora.eval()

        self.tokenizer.numeric_embedding.to(device)
        self.tokenizer.numeric_heads.to(device)

        text_ids, _ = self.tokenizer.encode_prefix(
            task_description=task_description,
            provenance_token=provenance_token,
        )
        text_ids = text_ids.to(device)

        states = torch.from_numpy(trajectory_prefix["states"]).float().unsqueeze(0).to(device)
        actions = torch.from_numpy(trajectory_prefix["actions"]).float().unsqueeze(0).to(device)
        rtgs = torch.from_numpy(trajectory_prefix["rtg"]).float().unsqueeze(0).to(device)
        if rtgs.ndim == 2:
            rtgs = rtgs.unsqueeze(-1)

        T = states.shape[1]

        combined, attn, L_text = self.tokenizer.build_dt_sequence(
            self.llm.model.get_input_embeddings(),
            text_ids, rtgs, states, actions, device,
        )
        out = self.llm.model(
            inputs_embeds=combined.to(dtype=model_dtype),
            attention_mask=attn,
            output_hidden_states=True,
        )
        hidden = out.hidden_states[-1].float()

        pos_last_s = L_text + 3 * (T - 1) + 1
        h = hidden[:, pos_last_s, :]
        pred_action = self.tokenizer.numeric_heads.predict_action(h)

        return pred_action.squeeze(0).cpu().numpy()
