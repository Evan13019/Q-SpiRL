# QL.py
import torch
import torch.nn as nn
import numpy as np
import random

class Q_Table(nn.Module):
    # Q-table state space:
    #     [R_o, D_o, R_T, A_T_o]
    # and 5 possible actions.
    # R_o   in {1..8}
    # D_o   in {0..4}
    # R_T   in {1..8}
    # A_T_o in {1..8}

    def __init__(self,
                 n_R=8,    
                 n_D=5,    
                 n_RT=8,   
                 n_A=8,    
                 n_actions=5):
        super().__init__()
        self.n_R = n_R
        self.n_D = n_D
        self.n_RT = n_RT
        self.n_A = n_A
        self.n_actions = n_actions

        # Q-table shape: (n_R, n_D, n_RT, n_A, n_actions)
        table_dim = (n_R, n_D, n_RT, n_A, n_actions)

        self.table = nn.Parameter(torch.zeros(table_dim, dtype=torch.float32))

    def _encode_state(self, state):
        # Converts raw env state [R_o, D_o, R_T, A_T_o] into
        # indices for the Q-table.
        if isinstance(state, np.ndarray):
            state = state.astype(int).tolist()
        elif isinstance(state, torch.Tensor):
            state = state.long().tolist()
        elif isinstance(state, (list, tuple)):
            state = list(state)
        else:
            raise ValueError(f"Unexpected state type: {type(state)}")

        assert len(state) == 4, f"Expected state of length 4, got {len(state)}"

        R_o, D_o, R_T, A_T_o = [int(x) for x in state]
        R_o_idx  = R_o  - 1          # 1..8 -> 0..7
        D_o_idx  = D_o              # 0..4 -> 0..4
        R_T_idx  = R_T  - 1          # 1..8 -> 0..7
        A_T_idx  = A_T_o - 1         # 1..8 -> 0..7

        # safety checks
        assert 0 <= R_o_idx < self.n_R
        assert 0 <= D_o_idx < self.n_D
        assert 0 <= R_T_idx < self.n_RT
        assert 0 <= A_T_idx < self.n_A

        return (R_o_idx, D_o_idx, R_T_idx, A_T_idx)

    def _get_q_values(self, state):
        """
        Return Q(s, :) as 1D tensor of size (n_actions,)
        """
        idx = self._encode_state(state)
        q_vals = self.table[idx]  
        return q_vals.view(-1)

    def forward(self, state):
        """
        Forward pass returns Q-values for given state: shape (n_actions,).
        """
        return self._get_q_values(state)

    def act(self, state, epsilon=0.0):
        """
        epsilon-greedy action selection - epsilon set to 0 for inference
        """
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        else:
            q_vals = self._get_q_values(state)
            return int(torch.argmax(q_vals).item())
        
    @torch.no_grad()
    def load_from_model(self, model, enc, device="cpu", num_samples=10):
        """
        Convert QSNN to Q-table by averaging over multiple random spike encodings.
        Args:
            model: Trained QSNN model
            enc: Frequency encoder (with randomness)
            device: Device to run inference on
            num_samples: Number of random spike patterns to average over (default: 10)
                        Higher = more stable but slower conversion (for SNN only)
        """
        model = model.to(device).eval()
        self.table.data.zero_()   
        mlp=False
        if num_samples == 0:
            mlp=True
            num_samples=1

        print(f"Converting QSNN to Q-table (averaging over {num_samples} samples per state)...")
        total_states = self.n_R * self.n_D * self.n_RT * self.n_A
        state_count = 0

        for R_o in range(1, self.n_R + 1):
            for D_o in range(0, self.n_D):
                for R_T in range(1, self.n_RT + 1):
                    for A_T_o in range(1, self.n_A + 1):
                        state_count += 1

                        state = np.array([R_o, D_o, R_T, A_T_o], dtype=np.float32)
                        # Average Q-values over multiple random encodings
                        q_vals_sum = torch.zeros(self.n_actions, device=device)
                        for _ in range(num_samples):
                            if mlp:
                                x = enc(state)
                            else:
                                spikes_np = enc.encode_state(state)
                                x = torch.tensor(spikes_np, dtype=torch.float32).to(device)
                            q_vals = model(x)

                            # Normalize output shape
                            if q_vals.dim() == 2:
                                q_vals = q_vals.squeeze(0)

                            q_vals_sum += q_vals

                        # Average the Q-values
                        q_vals_avg = q_vals_sum / num_samples

                        idx = self._encode_state(state)
                        self.table.data[idx] = q_vals_avg.detach().cpu()

                        if state_count % 500 == 0:
                            print(f"  Processed {state_count}/{total_states} states...")
