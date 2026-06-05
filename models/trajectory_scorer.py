                             

from typing import Any, Dict, List

import math
import numpy as np
import torch

from models.objective_heads import _forward_dt

class TrajectoryScorer:
    def __init__(self, llm, adapters, tokenizer, use_objectives: List[str]):
        self.llm = llm
        self.adapters = adapters
        self.tokenizer = tokenizer
        self.use_objectives = use_objectives

    @staticmethod
    def _gaussian_log_likelihood(x: torch.Tensor, mean: torch.Tensor,
                                 sigma: float = 1.0) -> torch.Tensor:
        
        var = sigma ** 2
        return -0.5 * ((x - mean) ** 2) / var - 0.5 * math.log(2 * math.pi * var)

    def _activate_lora(self, objective_name: str):
        
        if self.adapters is None:
            return
        self.adapters.activate(objective_name)
        obj_adapter, shared_adapter = self.adapters.get_active_adapters(objective_name)
        for module in (obj_adapter, shared_adapter):
            if module is not None:
                module.eval()

    def _prepare_trajectory(self, traj: Dict[str, Any], task_description: str,
                            device: torch.device, reverse: bool = False):
        
        states = torch.from_numpy(traj["states"]).float()            
        actions = torch.from_numpy(traj["actions"]).float()           
        rewards = torch.from_numpy(traj["rewards"]).float()           
        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(-1)

        if reverse:
            states = states.flip(0)
            actions = actions.flip(0)
            rewards = rewards.flip(0)

        T = states.shape[0]

        rtgs = torch.zeros_like(rewards)
        rtgs[-1] = rewards[-1]
        for t in range(T - 2, -1, -1):
            rtgs[t] = rewards[t] + rtgs[t + 1]

        states = states.unsqueeze(0).to(device)               
        actions = actions.unsqueeze(0).to(device)              
        rewards = rewards.unsqueeze(0).to(device)              
        rtgs = rtgs.unsqueeze(0).to(device)                    

        text_ids, _ = self.tokenizer.encode_prefix(
            task_description=task_description,
            provenance_token="augmented",
        )
        text_ids = text_ids.to(device)

        return text_ids, states, actions, rewards, rtgs

    def _rollout_loglikelihood(self, hidden, L_text, states, actions, rewards,
                               start_t=0, end_t=None):
        
        T = states.shape[1]
        if end_t is None:
            end_t = T

        nh = self.tokenizer.numeric_heads
        total_logp = 0.0
        count = 0

        for t in range(start_t, end_t):
            pos_R = L_text + 3 * t
            pos_s = pos_R + 1
            pos_a = pos_R + 2

            pred_s = nh.predict_next_state(hidden[:, pos_R, :])
            logp_s = self._gaussian_log_likelihood(states[:, t, :], pred_s)
            total_logp += logp_s.mean().item()

            pred_a = nh.predict_action(hidden[:, pos_s, :])
            logp_a = self._gaussian_log_likelihood(actions[:, t, :], pred_a)
            total_logp += logp_a.mean().item()

            pred_r = nh.predict_reward(hidden[:, pos_a, :])
            logp_r = self._gaussian_log_likelihood(rewards[:, t, :], pred_r)
            total_logp += logp_r.mean().item()

            count += 3

        return total_logp / max(count, 1)

    @torch.no_grad()
    def _score_O1(self, traj: Dict[str, Any], task_description: str) -> float:
        device = next(self.llm.model.parameters()).device
        self._activate_lora("O1")

        states = torch.from_numpy(traj["states"]).float()
        actions = torch.from_numpy(traj["actions"]).float()
        rewards = torch.from_numpy(traj["rewards"]).float().reshape(-1)
        T = states.shape[0]
        if T <= 1:
            return 0.0

        rtg_vals = torch.zeros(T)
        running = 0.0
        for t in range(T - 1, -1, -1):
            running += rewards[t].item()
            rtg_vals[t] = running

        B = T - 1
        s_t = states[:-1].unsqueeze(1).to(device)                
        a_t = actions[:-1].unsqueeze(1).to(device)                
        s_tp1 = states[1:].to(device)                          
        rtgs = rtg_vals[:-1].unsqueeze(1).unsqueeze(-1).to(device)             

        text_ids, _ = self.tokenizer.encode_prefix(
            task_description=task_description,
            provenance_token="augmented",
        )
        text_ids = text_ids.to(device)

        hidden, L_text, _ = _forward_dt(
            self.llm, self.tokenizer, text_ids, rtgs, s_t, a_t, device
        )

        h_last = hidden[:, -1, :]          
        pred_s = self.tokenizer.numeric_heads.predict_next_state(h_last)          
        logp = self._gaussian_log_likelihood(s_tp1, pred_s)
        return float(logp.mean().item())

    @torch.no_grad()
    def _score_O2(self, traj: Dict[str, Any], task_description: str) -> float:
        device = next(self.llm.model.parameters()).device
        self._activate_lora("O2")

        text_ids, states, actions, rewards, rtgs = self._prepare_trajectory(
            traj, task_description, device
        )
        T = states.shape[1]
        if T <= 2:
            return 0.0

        hidden, L_text, _ = _forward_dt(
            self.llm, self.tokenizer, text_ids, rtgs, states, actions, device
        )

        start_t = T // 4
        end_t = T - T // 4
        if end_t <= start_t:
            end_t = min(start_t + 1, T)

        return self._rollout_loglikelihood(
            hidden, L_text, states, actions, rewards, start_t, end_t
        )

    @torch.no_grad()
    def _score_O3(self, traj: Dict[str, Any], task_description: str) -> float:
        device = next(self.llm.model.parameters()).device
        self._activate_lora("O3")

        text_ids, states, actions, rewards, rtgs = self._prepare_trajectory(
            traj, task_description, device
        )
        if states.shape[1] <= 1:
            return 0.0

        hidden, L_text, _ = _forward_dt(
            self.llm, self.tokenizer, text_ids, rtgs, states, actions, device
        )
        return self._rollout_loglikelihood(hidden, L_text, states, actions, rewards)

    @torch.no_grad()
    def _score_O4(self, traj: Dict[str, Any], task_description: str) -> float:
        device = next(self.llm.model.parameters()).device
        self._activate_lora("O4")

        text_ids, states, actions, rewards, rtgs = self._prepare_trajectory(
            traj, task_description, device, reverse=True
        )
        if states.shape[1] <= 1:
            return 0.0

        hidden, L_text, _ = _forward_dt(
            self.llm, self.tokenizer, text_ids, rtgs, states, actions, device
        )
        return self._rollout_loglikelihood(hidden, L_text, states, actions, rewards)

    @torch.no_grad()
    def _score_O5(self, traj: Dict[str, Any], task_description: str) -> float:
        
        device = next(self.llm.model.parameters()).device
        self._activate_lora("O5")

        states = torch.from_numpy(traj["states"]).float()
        actions = torch.from_numpy(traj["actions"]).float()
        rewards = torch.from_numpy(traj["rewards"]).float()
        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(-1)
        T = states.shape[0]

        rtgs = torch.zeros_like(rewards)
        rtgs[-1] = rewards[-1]
        for t in range(T - 2, -1, -1):
            rtgs[t] = rewards[t] + rtgs[t + 1]

        ne = self.tokenizer.numeric_embedding
        flat_r = rtgs.to(device)                     
        flat_s = states.to(device)                   
        flat_a = actions.to(device)                  
        e_r = ne.embed_return(flat_r)                
        e_s = ne.embed_state(flat_s)                 
        e_a = ne.embed_action(flat_a)                
                                                         
        numeric = torch.stack([e_r, e_s, e_a], dim=1).reshape(T * 3, -1)           
        numeric = numeric.unsqueeze(0).float()              

        text_ids, _ = self.tokenizer.encode_prefix(
            task_description=task_description,
            provenance_token="augmented",
        )
        text_ids = text_ids.to(device)
        word_embed_fn = self.llm.model.get_input_embeddings()
        text_embeds = word_embed_fn(text_ids).float()                  
        L_text = text_embeds.shape[1]

        model_dtype = next(self.llm.model.parameters()).dtype
        combined = torch.cat([numeric, text_embeds], dim=1).to(dtype=model_dtype)
        attn = torch.ones(1, combined.shape[1], device=device, dtype=torch.long)

        out = self.llm.model(
            inputs_embeds=combined,
            attention_mask=attn,
            output_hidden_states=False,
        )
        logits = out.logits.float()                       

        L_traj = T * 3
        text_logits = logits[:, L_traj:-1, :]                     
        text_targets = text_ids[:, 1:]                          

        log_probs = torch.log_softmax(text_logits, dim=-1)                    
        token_logp = log_probs.gather(2, text_targets.unsqueeze(-1)).squeeze(-1)                 
        return float(token_logp.mean().item())

    @torch.no_grad()
    def _score_O6(self, traj: Dict[str, Any], task_description: str) -> float:
        device = next(self.llm.model.parameters()).device
        self._activate_lora("O6")

        text_ids, states, actions, rewards, rtgs = self._prepare_trajectory(
            traj, task_description, device
        )
        if states.shape[1] <= 1:
            return 0.0

        hidden, L_text, _ = _forward_dt(
            self.llm, self.tokenizer, text_ids, rtgs, states, actions, device
        )
        return self._rollout_loglikelihood(hidden, L_text, states, actions, rewards)

    def score(self, trajectory: Dict[str, Any], task_description: str) -> float:
        dispatch = {
            "O1": self._score_O1,
            "O2": self._score_O2,
            "O3": self._score_O3,
            "O4": self._score_O4,
            "O5": self._score_O5,
            "O6": self._score_O6,
        }
        scores = []
        for obj_name in self.use_objectives:
            fn = dispatch.get(obj_name)
            if fn is not None:
                scores.append(fn(trajectory, task_description))

        if not scores:
            return 0.0
        return float(sum(scores) / len(scores))

    def score_batch(self, trajectories: List[Dict[str, Any]], task_description: str) -> List[float]:
        
        if not trajectories:
            return []

        dispatch = {
            "O1": self._score_O1_batch,
            "O2": self._score_O2_batch,
            "O3": self._score_O3_batch,
            "O4": self._score_O4_batch,
            "O5": self._score_O5_batch,
            "O6": self._score_O6_batch,
        }

        B = len(trajectories)
                                                              
        all_obj_scores = []
        for obj_name in self.use_objectives:
            fn = dispatch.get(obj_name)
            if fn is not None:
                obj_scores = fn(trajectories, task_description)                    
                all_obj_scores.append(obj_scores)

        if not all_obj_scores:
            return [0.0] * B

        result = []
        for i in range(B):
            s = sum(obj_scores[i] for obj_scores in all_obj_scores) / len(all_obj_scores)
            result.append(s)
        return result

    def _prepare_trajectory_batch(self, trajectories: List[Dict[str, Any]],
                                   task_description: str, device: torch.device,
                                   reverse: bool = False):
        
        B = len(trajectories)
        T = trajectories[0]["states"].shape[0]

        states = torch.stack([torch.from_numpy(t["states"]).float() for t in trajectories])              
        actions = torch.stack([torch.from_numpy(t["actions"]).float() for t in trajectories])             
        rewards_list = []
        for t in trajectories:
            r = torch.from_numpy(t["rewards"]).float()
            if r.ndim == 1:
                r = r.unsqueeze(-1)
            rewards_list.append(r)
        rewards = torch.stack(rewards_list)             

        if reverse:
            states = states.flip(1)
            actions = actions.flip(1)
            rewards = rewards.flip(1)

        rtgs = torch.zeros_like(rewards)
        rtgs[:, -1] = rewards[:, -1]
        for t_idx in range(T - 2, -1, -1):
            rtgs[:, t_idx] = rewards[:, t_idx] + rtgs[:, t_idx + 1]

        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        rtgs = rtgs.to(device)

        text_ids, _ = self.tokenizer.encode_prefix(
            task_description=task_description,
            provenance_token="augmented",
        )
        text_ids = text_ids.to(device)
                                  
        text_ids = text_ids.expand(B, -1)

        return text_ids, states, actions, rewards, rtgs

    def _rollout_loglikelihood_batch(self, hidden, L_text, states, actions, rewards,
                                     start_t=0, end_t=None):
        
        B = states.shape[0]
        T = states.shape[1]
        if end_t is None:
            end_t = T

        nh = self.tokenizer.numeric_heads
                               
        total_logp = torch.zeros(B, device=states.device)
        count = 0

        for t in range(start_t, end_t):
            pos_R = L_text + 3 * t
            pos_s = pos_R + 1
            pos_a = pos_R + 2

            pred_s = nh.predict_next_state(hidden[:, pos_R, :])
            logp_s = self._gaussian_log_likelihood(states[:, t, :], pred_s).mean(dim=-1)        
            total_logp += logp_s

            pred_a = nh.predict_action(hidden[:, pos_s, :])
            logp_a = self._gaussian_log_likelihood(actions[:, t, :], pred_a).mean(dim=-1)
            total_logp += logp_a

            pred_r = nh.predict_reward(hidden[:, pos_a, :])
            logp_r = self._gaussian_log_likelihood(rewards[:, t, :], pred_r).mean(dim=-1)
            total_logp += logp_r

            count += 3

        per_sample = (total_logp / max(count, 1)).tolist()
        return per_sample

    @torch.no_grad()
    def _score_O1_batch(self, trajectories: List[Dict[str, Any]], task_description: str) -> List[float]:

        self._activate_lora("O1")
        return [self._score_O1(t, task_description) for t in trajectories]

    @torch.no_grad()
    def _score_O2_batch(self, trajectories: List[Dict[str, Any]], task_description: str) -> List[float]:
        device = next(self.llm.model.parameters()).device
        self._activate_lora("O2")
        text_ids, states, actions, rewards, rtgs = self._prepare_trajectory_batch(
            trajectories, task_description, device
        )
        T = states.shape[1]
        if T <= 2:
            return [0.0] * len(trajectories)
        hidden, L_text, _ = _forward_dt(self.llm, self.tokenizer, text_ids, rtgs, states, actions, device)
        start_t = T // 4
        end_t = T - T // 4
        if end_t <= start_t:
            end_t = min(start_t + 1, T)
        return self._rollout_loglikelihood_batch(hidden, L_text, states, actions, rewards, start_t, end_t)

    @torch.no_grad()
    def _score_O3_batch(self, trajectories: List[Dict[str, Any]], task_description: str) -> List[float]:
        device = next(self.llm.model.parameters()).device
        self._activate_lora("O3")
        text_ids, states, actions, rewards, rtgs = self._prepare_trajectory_batch(
            trajectories, task_description, device
        )
        if states.shape[1] <= 1:
            return [0.0] * len(trajectories)
        hidden, L_text, _ = _forward_dt(self.llm, self.tokenizer, text_ids, rtgs, states, actions, device)
        return self._rollout_loglikelihood_batch(hidden, L_text, states, actions, rewards)

    @torch.no_grad()
    def _score_O4_batch(self, trajectories: List[Dict[str, Any]], task_description: str) -> List[float]:
        device = next(self.llm.model.parameters()).device
        self._activate_lora("O4")
        text_ids, states, actions, rewards, rtgs = self._prepare_trajectory_batch(
            trajectories, task_description, device, reverse=True
        )
        if states.shape[1] <= 1:
            return [0.0] * len(trajectories)
        hidden, L_text, _ = _forward_dt(self.llm, self.tokenizer, text_ids, rtgs, states, actions, device)
        return self._rollout_loglikelihood_batch(hidden, L_text, states, actions, rewards)

    @torch.no_grad()
    def _score_O5_batch(self, trajectories: List[Dict[str, Any]], task_description: str) -> List[float]:
        
        device = next(self.llm.model.parameters()).device
        self._activate_lora("O5")
        B = len(trajectories)

        T = trajectories[0]["states"].shape[0]
        ne = self.tokenizer.numeric_embedding

        all_numeric = []
        for traj in trajectories:
            states_t = torch.from_numpy(traj["states"]).float().to(device)
            actions_t = torch.from_numpy(traj["actions"]).float().to(device)
            rewards_t = torch.from_numpy(traj["rewards"]).float().to(device)
            if rewards_t.ndim == 1:
                rewards_t = rewards_t.unsqueeze(-1)
                         
            rtgs_t = torch.zeros_like(rewards_t)
            rtgs_t[-1] = rewards_t[-1]
            for t_idx in range(T - 2, -1, -1):
                rtgs_t[t_idx] = rewards_t[t_idx] + rtgs_t[t_idx + 1]

            e_r = ne.embed_return(rtgs_t)
            e_s = ne.embed_state(states_t)
            e_a = ne.embed_action(actions_t)
            numeric = torch.stack([e_r, e_s, e_a], dim=1).reshape(T * 3, -1)           
            all_numeric.append(numeric)

        numeric_batch = torch.stack(all_numeric).float()              

        text_ids, _ = self.tokenizer.encode_prefix(
            task_description=task_description,
            provenance_token="augmented",
        )
        text_ids = text_ids.to(device)
        word_embed_fn = self.llm.model.get_input_embeddings()
        text_embeds = word_embed_fn(text_ids).float().expand(B, -1, -1)                  
        L_text = text_embeds.shape[1]

        model_dtype = next(self.llm.model.parameters()).dtype
        combined = torch.cat([numeric_batch, text_embeds], dim=1).to(dtype=model_dtype)
        attn = torch.ones(B, combined.shape[1], device=device, dtype=torch.long)

        out = self.llm.model(
            inputs_embeds=combined,
            attention_mask=attn,
            output_hidden_states=False,
        )
        logits = out.logits.float()                       

        L_traj = T * 3
        text_logits = logits[:, L_traj:-1, :]                             
        text_targets = text_ids.expand(B, -1)[:, 1:]                   

        log_probs = torch.log_softmax(text_logits, dim=-1)
        token_logp = log_probs.gather(2, text_targets.unsqueeze(-1)).squeeze(-1)                 
        return token_logp.mean(dim=-1).tolist()

    @torch.no_grad()
    def _score_O6_batch(self, trajectories: List[Dict[str, Any]], task_description: str) -> List[float]:
        device = next(self.llm.model.parameters()).device
        self._activate_lora("O6")
        text_ids, states, actions, rewards, rtgs = self._prepare_trajectory_batch(
            trajectories, task_description, device
        )
        if states.shape[1] <= 1:
            return [0.0] * len(trajectories)
        hidden, L_text, _ = _forward_dt(self.llm, self.tokenizer, text_ids, rtgs, states, actions, device)
        return self._rollout_loglikelihood_batch(hidden, L_text, states, actions, rewards)
