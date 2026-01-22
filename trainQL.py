# =========================
# train_dsqn.py (with Q-Table support)
# =========================
import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from env import GridEnvCells
from models.vanilla_models import QMLP, QSNN   
from models.quantum_models import HybridMLP, HybridSNN     
from models.Qtable import Q_Table
from helper.encode import freq_encoder, encode_one_hot
from helper.prints import print_config, print_model_summary, print_encoder_info, print_training_header
from helper.misc import ReplayBuffer, parse_args
from helper.plots import plot_metrics
import random


def pick_model_class(qmlp: bool, quantum: bool, qtable: bool):
    if qtable:
        return Q_Table
    if qmlp:
        return HybridMLP if quantum else QMLP
    else:
        return HybridSNN if quantum else QSNN

args = parse_args()
# if args.qmlp exit program (keep original behavior unless qtable)
if args.qmlp and not args.qtable:
    exit()

# ---- Print configuration ----
print_config(args)

# ---- seeds ----
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
print(f"Using device: {device}")

# Always one shared directory for checkpoints
os.makedirs(args.model_dir, exist_ok=True)

# ---- env ----
env_params = {
    "num_static": args.num_static,
    "num_dynamic": args.num_dynamic,
    "seed": args.env_seed,
}
env_ = GridEnvCells(**env_params)

# ---- model ----
mod = pick_model_class(qmlp=args.qmlp, quantum=args.quantum, qtable=args.qtable)

if not args.qtable:
    effective_rate = args.max_freq * args.dt * args.num_timesteps
    print(f"[Encoder] Effective spikes per channel ~ {effective_rate:.2f}")

# Build model based on type
if args.qtable:
    # Q-table: uses raw state indices, no neural network
    net_params = {
        "n_R": 8,      # R_o bins (1-8)
        "n_D": 5,      # D_o bins (0-4)
        "n_RT": 8,     # R_T bins (1-8)
        "n_A": 8,      # A_T_o bins (1-8)
        "n_actions": args.num_actions,
    }
    print(f"\n[Model] Using Q-Table (tabular Q-learning)")
    print(f"[Model] State space: {8*5*8*8} = 2560 states")
    print(f"[Model] Parameters: {net_params}")
    
    net_ = mod(**net_params)
    target_net = None  # Q-table doesn't use target network
else:
    net_params = {
        "input_size": args.input_size,
        "hidden_size": 30 if not args.quantum else 35,
        "num_layers": 3,
        "num_actions": args.num_actions,
        "qubits": args.qubits,
    }
    # add mem_tau and syn_tau for SNNs
    if not args.qmlp:
        net_params["tau_mem"] = args.mem_tau
        net_params["tau_syn"] = args.syn_tau
        net_params["type"] = args.type

    print(f"\n[Model] Using {mod.__name__}")
    print(f"[Model] Parameters: {net_params}")

    net_ = mod(**net_params).to(device)
    target_net = mod(**net_params).to(device)
    target_net.load_state_dict(net_.state_dict())
    target_net.eval()

# ---- Print model summary ----
if args.qtable:
    total_params = 8 * 5 * 8 * 8 * args.num_actions
    print(f"\n[Model Summary] Q-Table")
    print(f"  Table shape: (8, 5, 8, 8, {args.num_actions})")
    print(f"  Total entries: {total_params:,}")
else:
    print_model_summary(net_, name="Policy Network")

if args.verbose and not args.qtable:
    print(f"\n[Architecture]\n{net_}")

# ---- encoder (only for neural network models) ----
enc = None
if not args.qtable:
    encoder_params = {
        "max_freq": args.max_freq,
        "num_timesteps": args.num_timesteps,
        "dt": args.dt,
        "bg_freq": args.bg_freq,
        "noise_std": args.noise_std,
        "seed": args.enc_seed,
    }
    enc = freq_encoder(**encoder_params)

    if not args.qmlp:
        print_encoder_info(enc)
        # Print spike probability for sanity check
        spike_prob = 1.0 - np.exp(-args.max_freq * args.dt)
        print(f"[Encoder] Spike probability at max_freq: {spike_prob:.3f}")

# ---- optimizer / replay ----
if args.qtable:
    optimizer = None  # Q-table uses direct updates
    replay_buffer = None  # Q-table uses online learning
else:
    optimizer = torch.optim.Adam(net_.parameters(), lr=args.lr)
    replay_buffer = ReplayBuffer(capacity=args.buffer_capacity)

# ---- bookkeeping ----
rewards, losses, final_dists, episode_lengths = [], [], [], []
goal_flags, collision_flags, oob_flags, timeout_flags = [], [], [], []

# Track collision types separately
static_collision_flags = []
dynamic_collision_flags = []

tag = f"qtable={args.qtable}_qmlp={args.qmlp}_quantum={args.quantum}_qubits={args.qubits}_T={args.num_timesteps}_maxFreq={args.max_freq}_dt={args.dt}_seed={args.seed}_ns={args.num_static}_nd={args.num_dynamic}"
last_model_path = os.path.join(args.model_dir, f"last_{tag}.pth")

print(f"\n[Checkpoints]")
print(f"  Last model: {last_model_path}")

# Each configuration gets its own plot directory (only if plots enabled)
plot_dir = None
if args.plots:
    plot_dir = os.path.join(args.plot_root, tag)
    os.makedirs(plot_dir, exist_ok=True)
    print(f"  Plot dir:   {plot_dir}")

global_step = 0
T = args.T0

# ---- Print training header ----
print_training_header()

for ep in range(args.episodes):
    state_raw = env_.reset(seed=ep).astype(np.float32)

    # Encode state based on model type
    if args.qtable:
        # Q-table uses raw state directly
        state_feat = state_raw
    elif not args.qmlp:
        state_feat = enc.encode_state(state_raw)
    else:
        state_feat = encode_one_hot(state_raw)

    episode_reward = 0.0
    episode_loss = 0.0
    num_updates = 0
    done = False
    steps_in_ep = 0
    last_info = {}

    while not done:
        global_step += 1
        steps_in_ep += 1

        # temperature schedule
        T = max(args.T_min, args.T0 * (args.lambda_T ** global_step))

        # Get Q-values based on model type
        if args.qtable:
            q_values = net_(state_feat).detach().cpu().numpy()
        else:
            s_torch = torch.tensor(state_feat, dtype=torch.float32, device=device)
            with torch.no_grad():
                q_values = net_(s_torch).squeeze().detach().cpu().numpy()

        valid_mask = env_.get_valid_actions()
        valid_actions = np.where(valid_mask)[0]

        # hybrid exploration: epsilon random valid, else Boltzmann over valid
        if len(valid_actions) == 0:
            action = np.random.randint(0, args.num_actions)
        else:
            if np.random.rand() < args.epsilon:
                action = int(np.random.choice(valid_actions))
            else:
                q_valid = q_values[valid_actions]
                if len(q_valid) == 1:
                    action = int(valid_actions[0])
                else:
                    q_stable = q_valid - np.max(q_valid)
                    exp_q = np.exp(q_stable / T)
                    probs = exp_q / (exp_q.sum() + 1e-10)
                    action = int(np.random.choice(valid_actions, p=probs))

        # step
        next_state_raw, reward, done, info = env_.step(action)
        last_info = info
        episode_reward += float(reward)

        # Encode next state based on model type
        if args.qtable:
            next_state_feat = next_state_raw.astype(np.float32)
        elif not args.qmlp:
            next_state_feat = enc.encode_state(next_state_raw.astype(np.float32))
        else:
            next_state_feat = encode_one_hot(next_state_raw.astype(np.float32))

        # Learning update
        if args.qtable:
            # Tabular Q-learning update (online, no replay buffer)
            with torch.no_grad():
                # Get current Q-value
                current_q = net_(state_feat)[action].item()
                
                # Get max Q-value for next state
                if done:
                    max_next_q = 0.0
                else:
                    next_q_values = net_(next_state_feat)
                    max_next_q = next_q_values.max().item()
                
                # TD target
                td_target = reward + args.gamma * max_next_q
                
                # TD error
                td_error = td_target - current_q
                
                # Update Q-table
                idx = net_._encode_state(state_feat)
                net_.table.data[idx][action] += args.lr * td_error
                
                episode_loss += abs(td_error)
                num_updates += 1
        else:
            # Neural network learning with replay buffer
            replay_buffer.push(state_feat, action, reward, next_state_feat, done)

            if len(replay_buffer) >= max(args.batch_size, args.learn_start):
                states_b, actions_b, rewards_b, next_states_b, dones_b = replay_buffer.sample(args.batch_size)

                states_b = torch.as_tensor(states_b, dtype=torch.float32, device=device)
                next_states_b = torch.as_tensor(next_states_b, dtype=torch.float32, device=device)
                actions_b = torch.as_tensor(actions_b, dtype=torch.int64, device=device)
                rewards_b = torch.as_tensor(rewards_b, dtype=torch.float32, device=device)
                dones_b = torch.as_tensor(dones_b, dtype=torch.float32, device=device)

                q_all = net_(states_b)  # (B, A)
                q_pred = q_all.gather(1, actions_b.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    q_next_all = target_net(next_states_b)
                    max_next_q = q_next_all.max(dim=1)[0]
                    targets = rewards_b + args.gamma * max_next_q * (1.0 - dones_b)

                loss = F.smooth_l1_loss(q_pred, targets)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net_.parameters(), max_norm=1.0)
                optimizer.step()

                if global_step % args.target_update_freq == 0:
                    target_net.load_state_dict(net_.state_dict())

                episode_loss += float(loss.item())
                num_updates += 1

        state_feat = next_state_feat

    # end episode stats
    dxdy = env_.terminal_cell - env_.pos
    final_dist = float(np.linalg.norm(dxdy, ord=2))

    rewards.append(float(episode_reward))
    losses.append(float(episode_loss / max(1, num_updates)))
    final_dists.append(final_dist)
    episode_lengths.append(int(steps_in_ep))

    goal_flags.append(1 if last_info.get("goal", False) else 0)
    collision_flags.append(1 if last_info.get("collision", False) else 0)
    oob_flags.append(1 if last_info.get("out_of_bounds", False) else 0)
    timeout_flags.append(1 if last_info.get("timeout", False) else 0)
    
    # Track collision types
    collision_type = last_info.get("collision_type", None)
    static_collision_flags.append(1 if collision_type == "static" else 0)
    dynamic_collision_flags.append(1 if collision_type in ["dynamic", "dynamic_into_agent"] else 0)

    if ep % 10 == 0:
        recent_success = np.mean(goal_flags[-10:]) if len(goal_flags) >= 10 else np.mean(goal_flags)
        recent_collision = np.mean(collision_flags[-10:]) if len(collision_flags) >= 10 else np.mean(collision_flags)
        buffer_size = 0 if args.qtable else len(replay_buffer)
        print(
            f"{ep:>6} | {episode_reward:>10.2f} | {losses[-1]:>10.5f} | {final_dist:>8.2f} | "
            f"{T:>8.4f} | {steps_in_ep:>6} | {buffer_size:>8} | {recent_success:>7.1%} | {recent_collision:>7.1%}"
        )

# ---- Training complete ----
print(f"{'-'*110}")
print(f"{'='*110}")
print(f" Training Complete")
print(f"{'='*110}")

# save last
torch.save(net_.state_dict(), last_model_path)
print(f"  Last model saved:  {last_model_path}")
# Final statistics
print(f"\n  [Final Statistics]")
print(f"    Total episodes:     {args.episodes}")
print(f"    Total steps:        {global_step:,}")
print(f"    Final success rate: {np.mean(goal_flags[-100:]):.1%} (last 100 eps)")
print(f"    Final collision rate: {np.mean(collision_flags[-100:]):.1%} (last 100 eps)")
print(f"      - Static collisions:  {np.mean(static_collision_flags[-100:]):.1%}")
print(f"      - Dynamic collisions: {np.mean(dynamic_collision_flags[-100:]):.1%}")
print(f"    Final avg reward:   {np.mean(rewards[-100:]):.2f} (last 100 eps)")
print(f"    Final avg loss:     {np.mean(losses[-100:]):.5f} (last 100 eps)")
print(f"    Final avg steps:    {np.mean(episode_lengths[-100:]):.1f} (last 100 eps)")

if args.plots and plot_dir is not None:
    plot_metrics(
        rewards, losses, final_dists, goal_flags, collision_flags, oob_flags, timeout_flags,
        static_collision_flags, dynamic_collision_flags, episode_lengths, plot_dir
    )