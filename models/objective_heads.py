
import random

import torch
import torch.nn.functional as F

def extract_trajectory_window(batch, window_size, device):
    obs_full = batch["obs_full"]
    act_full = batch["act_full"]
    rew_full = batch["rew_full"]

    if isinstance(obs_full, torch.Tensor):
        B = obs_full.shape[0]
    else:
        B = len(obs_full)

    states_list, actions_list, rewards_list = [], [], []
    for i in range(B):
        if isinstance(obs_full, torch.Tensor):
            obs_i = obs_full[i].float()
            act_i = act_full[i].float()
            rew_i = rew_full[i].float()
        else:
            obs_i = torch.as_tensor(obs_full[i], dtype=torch.float32)
            act_i = torch.as_tensor(act_full[i], dtype=torch.float32)
            rew_i = torch.as_tensor(rew_full[i], dtype=torch.float32)

        T_i = obs_i.shape[0]
        W = min(window_size, T_i)
        start = random.randint(0, max(T_i - W, 0))

        s_w = obs_i[start : start + W]
        a_w = act_i[start : start + W]
        r_w = rew_i[start : start + W]
        if r_w.ndim == 1:
            r_w = r_w.unsqueeze(-1)

        states_list.append(s_w)
        actions_list.append(a_w)
        rewards_list.append(r_w)

    states = torch.stack(states_list).to(device)
    actions = torch.stack(actions_list).to(device)
    rewards = torch.stack(rewards_list).to(device)

    W = states.shape[1]
    rtgs = torch.zeros_like(rewards)
    rtgs[:, -1] = rewards[:, -1]
    for t in range(W - 2, -1, -1):
        rtgs[:, t] = rewards[:, t] + rtgs[:, t + 1]

    return states, actions, rewards, rtgs

def _autoregressive_loss(hidden, tokenizer, L_text, states, actions, rewards,
                         start_t=0, end_t=None):
    T = states.shape[1]
    if end_t is None:
        end_t = T

    loss = torch.tensor(0.0, device=hidden.device)
    count = 0

    for t in range(start_t, end_t):
        pos_R = L_text + 3 * t
        pos_s = pos_R + 1
        pos_a = pos_R + 2

        pred_s = tokenizer.numeric_heads.predict_next_state(hidden[:, pos_R, :])
        loss = loss + F.mse_loss(pred_s, states[:, t, :])

        pred_a = tokenizer.numeric_heads.predict_action(hidden[:, pos_s, :])
        loss = loss + F.mse_loss(pred_a, actions[:, t, :])

        pred_r = tokenizer.numeric_heads.predict_reward(hidden[:, pos_a, :])
        loss = loss + F.mse_loss(pred_r, rewards[:, t, :])

        count += 3

    return loss / max(count, 1)

def _forward_dt(llm, tokenizer, text_ids, rtgs, states, actions, device):
    model_dtype = next(llm.model.parameters()).dtype
    combined, attn, L_text = tokenizer.build_dt_sequence(
        llm.model.get_input_embeddings(),
        text_ids, rtgs, states, actions, device,
    )
    combined = combined.to(dtype=model_dtype)
    out = llm.model(
        inputs_embeds=combined,
        attention_mask=attn,
        output_hidden_states=True,
    )
    hidden = out.hidden_states[-1].float()
    return hidden, L_text, out

def compute_loss_O1(llm, tokenizer, text_ids, batch, device, **_kw):
    obs_t   = batch["obs_t"].float().to(device)
    act_t   = batch["act_t"].float().to(device)
    obs_tp1 = batch["obs_tp1"].float().to(device)
    B = obs_t.shape[0]

    rtgs    = torch.zeros(B, 1, 1, device=device)
    states  = obs_t.unsqueeze(1)
    actions = act_t.unsqueeze(1)

    hidden, L_text, _ = _forward_dt(llm, tokenizer, text_ids, rtgs, states, actions, device)

    h_last = hidden[:, -1, :]
    pred_s = tokenizer.numeric_heads.predict_next_state(h_last)
    return F.mse_loss(pred_s, obs_tp1)

def compute_loss_O2(llm, tokenizer, text_ids, batch, device, window_size=20, **_kw):
    states, actions, rewards, rtgs = extract_trajectory_window(batch, window_size, device)
    T = states.shape[1]

    hidden, L_text, _ = _forward_dt(llm, tokenizer, text_ids, rtgs, states, actions, device)

    start_t = T // 4
    end_t = T - T // 4
    if end_t <= start_t:
        end_t = min(start_t + 1, T)

    return _autoregressive_loss(hidden, tokenizer, L_text, states, actions, rewards, start_t, end_t)

def compute_loss_O3(llm, tokenizer, text_ids, batch, device, window_size=20, **_kw):
    states, actions, rewards, rtgs = extract_trajectory_window(batch, window_size, device)
    hidden, L_text, _ = _forward_dt(llm, tokenizer, text_ids, rtgs, states, actions, device)
    return _autoregressive_loss(hidden, tokenizer, L_text, states, actions, rewards)

def compute_loss_O4(llm, tokenizer, text_ids, batch, device, window_size=20, **_kw):
    states, actions, rewards, rtgs = extract_trajectory_window(batch, window_size, device)

    states  = states.flip(1)
    actions = actions.flip(1)
    rewards = rewards.flip(1)

    T = states.shape[1]
    rtgs = torch.zeros_like(rewards)
    rtgs[:, -1] = rewards[:, -1]
    for t in range(T - 2, -1, -1):
        rtgs[:, t] = rewards[:, t] + rtgs[:, t + 1]

    hidden, L_text, _ = _forward_dt(llm, tokenizer, text_ids, rtgs, states, actions, device)
    return _autoregressive_loss(hidden, tokenizer, L_text, states, actions, rewards)

def compute_loss_O5(llm, tokenizer, text_ids, batch, device, window_size=20, **_kw):
    states, actions, rewards, rtgs = extract_trajectory_window(batch, window_size, device)
    B, T = states.shape[0], states.shape[1]

    word_embed_fn = llm.model.get_input_embeddings()
    model_dtype = next(llm.model.parameters()).dtype

    flat_r = rtgs.reshape(B * T, 1)
    flat_s = states.reshape(B * T, -1)
    flat_a = actions.reshape(B * T, -1)
    e_r = tokenizer.numeric_embedding.embed_return(flat_r).reshape(B, T, -1)
    e_s = tokenizer.numeric_embedding.embed_state(flat_s).reshape(B, T, -1)
    e_a = tokenizer.numeric_embedding.embed_action(flat_a).reshape(B, T, -1)
    numeric = torch.stack([e_r, e_s, e_a], dim=2).reshape(B, T * 3, -1).float()

    text_embeds = word_embed_fn(text_ids.to(device)).float()
    L_text = text_embeds.shape[1]
    if B > 1:
        text_embeds = text_embeds.expand(B, -1, -1)
    text_ids_exp = text_ids.expand(B, -1).to(device)

    combined = torch.cat([numeric, text_embeds], dim=1).to(dtype=model_dtype)
    attn = torch.ones(B, combined.shape[1], device=device, dtype=torch.long)

    out = llm.model(inputs_embeds=combined, attention_mask=attn, output_hidden_states=False)
    logits = out.logits.float()

    L_traj = T * 3
    text_logits  = logits[:, L_traj:-1, :]
    text_targets = text_ids_exp[:, 1:]

    return F.cross_entropy(
        text_logits.reshape(-1, text_logits.shape[-1]),
        text_targets.reshape(-1),
    )

def compute_loss_O6(llm, tokenizer, text_ids, batch, device, window_size=20, **_kw):
    return compute_loss_O3(llm, tokenizer, text_ids, batch, device, window_size=window_size)

OBJECTIVE_FN = {
    "O1": compute_loss_O1,
    "O2": compute_loss_O2,
    "O3": compute_loss_O3,
    "O4": compute_loss_O4,
    "O5": compute_loss_O5,
    "O6": compute_loss_O6,
}
