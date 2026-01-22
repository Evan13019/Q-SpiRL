import torch
import torch.nn as nn
from norse.torch.functional.lif import LIFParameters
from collections import deque
import random
import numpy as np
from norse.torch.module.lif import LIFCell


class QSNN(nn.Module):
    def __init__(self, input_size=29, hidden_size=128, num_layers=3,
                 num_actions=5, qubits=None, tau_mem=0.020, tau_syn=0.010, type="max"):
        super().__init__()
        
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.type = type

        # LIF parameters: tau_syn = 10ms, tau_mem = 20ms, V_th = 1, V_r = 0
        self.lif_params = LIFParameters(
            tau_syn_inv=torch.tensor(1.0 / tau_syn),
            tau_mem_inv=torch.tensor(1.0 / tau_mem),
            v_leak=torch.tensor(0.0),
            v_th=torch.tensor(1.0),
            v_reset=torch.tensor(0.0),
            method="super",
            alpha=torch.tensor(100.0),
        )
        
        self.input_layer = nn.Linear(input_size, hidden_size, bias=True)
        self.input_lif = LIFCell(p=self.lif_params)
        self.hidden_layers = nn.ModuleList()
        self.hidden_lifs = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size, bias=True))
            self.hidden_lifs.append(LIFCell(p=self.lif_params))
        
        # Non-spiking output layer
        self.q_layer = nn.Linear(hidden_size, num_actions)
        print(f"Initialized QSNN with type='{self.type}'")
    
    def forward(self, x):
        # Detect input format and convert to (T, B, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_batch = True
        elif x.dim() == 3:
            x = x.permute(1, 0, 2)
            squeeze_batch = False
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")
        
        T, B, _ = x.shape
        device = x.device
        
        # Initialize LIF states
        input_state = None
        hidden_states = [None] * len(self.hidden_lifs)
        
        # For different aggregation methods
        if self.type == "mean":
            spike_accumulator = torch.zeros(B, self.hidden_size, device=device)
        elif self.type == "max":
            spike_max = torch.zeros(B, self.hidden_size, device=device)
        # For "last", we just use the final spikes

        # Process each timestep
        for t in range(T):
            # Input layer
            z = self.input_layer(x[t])
            spikes, input_state = self.input_lif(z, input_state)
            
            # Hidden layers
            for i, (linear, lif) in enumerate(zip(self.hidden_layers, self.hidden_lifs)):
                z = linear(spikes)
                spikes, hidden_states[i] = lif(z, hidden_states[i])
            
            # Aggregate spikes based on method
            if self.type == "mean":
                spike_accumulator = spike_accumulator + spikes
            elif self.type == "max":
                spike_max = torch.max(spike_max, spikes)
        
        # Compute firing rate based on aggregation type
        if self.type == "mean":
            # Accurate to paper
            firing_rate = spike_accumulator / T
        elif self.type == "max":
            # Max spike value across all timesteps
            firing_rate = spike_max
        elif self.type == "last":
            # Just use spikes from final timestep
            firing_rate = spikes
        else:
            raise ValueError(f"Unknown type: {self.type}")
        
        q_values = self.q_layer(firing_rate)
        
        if squeeze_batch:
            q_values = q_values.squeeze(0)
        
        return q_values

class QMLP(nn.Module):
    """
    Drop-in replacement for QSNN, but with standard linear layers + ReLU.
    """
    def __init__(self, input_size=29, hidden_size=128, num_layers=3, num_actions=5, qubits=None):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions

        layers = []

        # First layer: input_size -> hidden_size
        layers.append(nn.Linear(input_size, hidden_size, bias=True))
        layers.append(nn.ReLU())

        # Hidden layers: hidden_size -> hidden_size
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size, bias=True))
            layers.append(nn.ReLU())

        # Final linear: hidden_size -> num_actions
        layers.append(nn.Linear(hidden_size, num_actions, bias=True))

        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        # x: (input_size,) or (B, input_size)
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = x.float()

        if x.dim() == 1:
            x = x.unsqueeze(0)  

        q_values = self.net(x)       
        return q_values             