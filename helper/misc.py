import numpy as np
import torch
import time
import random
import random
from collections import deque
import argparse

PATH_SCALE = 24.6 / np.linalg.norm(np.array([28.0, 28.0]))  

def compute_path_length_m(positions):
    positions = np.array(positions, dtype=float)
    if len(positions) > 1:
        seg_lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
        path_length_grid = float(seg_lengths.sum())
    else:
        path_length_grid = 0.0
    return path_length_grid, path_length_grid * PATH_SCALE

def run_episode_qtable(env, qtab, max_steps=300, device='cpu'):
    qtab.eval()

    state = env.reset()  # discrete tuple / representation used by Q_Table

    positions = [env.pos.copy()]
    total_reward = 0.0
    steps = 0
    done = False
    last_info = {}

    start_time = time.perf_counter()

    while not done and steps < max_steps:
        valid_mask = env.get_valid_actions()

        with torch.no_grad():
            q_vals = qtab(state).cpu().numpy().squeeze()
            action = int(np.argmax(q_vals))

        next_state, reward, done, info = env.step(action)
        last_info = info
        total_reward += reward
        steps += 1
        positions.append(env.pos.copy())

        state = next_state

    end_time = time.perf_counter()
    computation_time = end_time - start_time

    path_length_grid, path_length_m = compute_path_length_m(positions)

    return {
        "total_reward": total_reward,
        "steps": steps,
        "path_length_grid": path_length_grid,
        "path_length_m": path_length_m,
        "computation_time": computation_time,
        "info": last_info,
    }

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state, copy=True),
            int(action),
            float(reward),
            np.array(next_state, copy=True),
            float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)
def parse_args():
    p = argparse.ArgumentParser()

    # model toggles
    p.add_argument("--qmlp", action="store_true", help="Use (Q)MLP-style model/encoding.")
    p.add_argument("--quantum", action="store_true", help="Use Hybrid* quantum variants.")
    p.add_argument("--qubits", type=int, default=3, help="Number of qubits for Hybrid* models.")
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--num_actions", type=int, default=5)
    p.add_argument("--mem_tau", type=float, default=0.02)
    p.add_argument("--syn_tau", type=float, default=0.01)
    p.add_argument("--type", type=str, default="max", choices=["mean", "max", "last"])
    
    # State encoding:
    # State = [R_o (1-8), D_o (0-4), R_T (1-8), A_T_o (1-8)]
    # One-hot sizes: 8 + 5 + 8 + 8 = 29
    p.add_argument("--input_size", type=int, default=29)

    # env
    p.add_argument("--num_static", type=int, default=10)
    p.add_argument("--num_dynamic", type=int, default=1)
    p.add_argument("--env_seed", type=int, default=1)

    # encoder (used when qmlp == False)
    # NOTE: With dt=0.01 and max_freq=80, spike_prob = 1 - exp(-80 * 0.01) ≈ 0.55
    # This gives reasonable spike density
    p.add_argument("--max_freq", type=float, default=100.0)
    p.add_argument("--num_timesteps", type=int, default=20)
    p.add_argument("--dt", type=float, default=0.01)  
    p.add_argument("--bg_freq", type=float, default=0.0)
    p.add_argument("--noise_std", type=float, default=0.0)
    p.add_argument("--enc_seed", type=int, default=42)

    # training
    p.add_argument("--episodes", type=int, default=800)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--lr", type=float, default=0.005)

    # exploration
    p.add_argument("--epsilon", type=float, default=0.01)       # random valid action prob
    p.add_argument("--T0", type=float, default=1.0)
    p.add_argument("--lambda_T", type=float, default=0.999)
    p.add_argument("--T_min", type=float, default=0.05)

    # replay buffer / DQN
    p.add_argument("--buffer_capacity", type=int, default=100000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--learn_start", type=int, default=1000)
    p.add_argument("--target_update_freq", type=int, default=500)

    # logging / saving
    p.add_argument("--model_dir", type=str, default="models", help="Where to save .pth checkpoints")
    p.add_argument("--plot_root", type=str, default="plots", help="Root folder for per-config plot directories")
    p.add_argument("--plots", action="store_true", help="Save training plots under plot_root/<config>/")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=123, help="Global seed for python/numpy/torch.")
    p.add_argument("--verbose", action="store_true", help="Print detailed model architecture.")

    return p.parse_args()
