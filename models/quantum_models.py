# If used in your project please cite this work as described in the README file.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import numpy as np
import torch
from torch import nn
import pennylane as qml
from norse.torch.functional.lif import LIFParameters
from norse.torch.module.lif import LIFCell

class HybridMLP(nn.Module):
    def __init__(self, input_size, num_actions, qubits, num_layers, ansatz='iqp', gates='rot', hidden_size=40):
        super().__init__()
        self.input_dim = input_size
        self.output_dim = num_actions
        layers = num_layers
        layers = [nn.Linear(self.input_dim, hidden_size, bias=True),
                  nn.ReLU(), nn.Linear(hidden_size, qubits, bias=True),
                  quantumNN(qubits, layers, ansatz=ansatz, gates=gates),
                  nn.Linear(qubits, hidden_size, bias=True),
                  nn.ReLU(),
                  nn.Linear(hidden_size, self.output_dim, bias=True)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # x: (input_size,) or (B, input_size)
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = x.float()

        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, F)

        q_values = self.model(x)          # (B, num_actions)
        return q_values  
    
class HybridSNN(nn.Module):
    """Matches DSQN paper architecture more closely"""
    def __init__(self, input_size, num_actions, qubits, num_layers, 
                 ansatz='iqp', gates='rot', hidden_size=40, tau_mem=0.02, tau_syn=0.01,
                 type="max"):
        super().__init__()
        
        self.qubits = qubits
        self.hidden_size = hidden_size
        self.type = type
        
        self.lif_params = LIFParameters(
            tau_syn_inv=torch.tensor(1.0 / tau_syn),
            tau_mem_inv=torch.tensor(1.0 / tau_mem),
            v_leak=torch.tensor(0.0),
            v_th=torch.tensor(1.0),
            v_reset=torch.tensor(0.0),
            method="super",
            alpha=torch.tensor(100.0),
        )
        
        # Spiking layers (LIF neurons)
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.lif1 = LIFCell(p=self.lif_params)
        
        self.fc2 = nn.Linear(hidden_size, qubits, bias=True)
        self.lif2 = LIFCell(p=self.lif_params)
        
        # Quantum layer
        self.qnn = quantumNN(qubits, num_layers, ansatz=ansatz, gates=gates)

        self.fc3 = nn.Linear(qubits, hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, num_actions, bias=True)
        print(f"HybridSNN initialized with type='{self.type}'")

    def forward(self, x):
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
        state1, state2 = None, None
        
        # Initialize based on aggregation type
        if self.type == "mean":
            pre_q_spike_sum = torch.zeros(B, self.qubits, device=device)
        elif self.type == "max":
            pre_q_spike_max = torch.zeros(B, self.qubits, device=device)
        # For "last", we just use final spikes

        # === Phase 1: Spiking layers BEFORE quantum ===
        for t in range(T):
            z1 = self.fc1(x[t])
            spikes1, state1 = self.lif1(z1, state1)
            
            z2 = self.fc2(spikes1)
            spikes2, state2 = self.lif2(z2, state2)
            
            # Aggregate based on type
            if self.type == "mean":
                pre_q_spike_sum = pre_q_spike_sum + spikes2
            elif self.type == "max":
                pre_q_spike_max = torch.max(pre_q_spike_max, spikes2)
        
        # === Phase 2: Compute firing rate based on aggregation ===
        if self.type == "mean":
            pre_q_rate = pre_q_spike_sum / T  # Values in [0, 1]
        elif self.type == "max":
            pre_q_rate = pre_q_spike_max
        elif self.type == "last":
            pre_q_rate = spikes2  # Final timestep spikes
        else:
            raise ValueError(f"Unknown type: {self.type}")
        
        # === Phase 3: Quantum layer on rate (continuous values) ===
        q_out = self.qnn(pre_q_rate)
        
        # === Phase 4: Post-quantum classical layers ===
        h3 = self.relu(self.fc3(q_out))
        q_values = self.output(h3)
        
        if squeeze_batch:
            q_values = q_values.squeeze(0)
            
        return q_values

def quantumNN(qubits, layers, ansatz, gates):

    dev = qml.device('default.qubit', wires=qubits)
    def qml_XYZ(a, b, c, qubit):
        qml.RX(a, wires=qubit)
        qml.RY(b, wires=qubit)
        qml.RZ(c, wires=qubit)

    def qml_CXYZ(a, b, c, control_qubit, target_qubit):
        qml.CRX(a, wires=[control_qubit, target_qubit])
        qml.CRY(b, wires=[control_qubit, target_qubit])
        qml.CRZ(c, wires=[control_qubit, target_qubit])

    def variational(inputs, scaling, weights, qubit, layer):
        if 'iqp' == ansatz:  # parameterized two-qubit rotations
            if 'rot' == gates:
                qml.CRot(scaling[layer, qubit, 0] * inputs[qubit] + weights[layer, qubit, 0],
                         scaling[layer, qubit, 1] * inputs[qubit] + weights[layer, qubit, 1],
                         scaling[layer, qubit, 2] * inputs[qubit] + weights[layer, qubit, 2],
                         wires=[qubit, (qubit + 1) % qubits])
            elif 'u3' == gates:
                qml.ctrl(qml.U3, qubit)(scaling[layer, qubit, 0] * inputs[qubit] + weights[layer, qubit, 0],
                                        scaling[layer, qubit, 1] * inputs[qubit] + weights[layer, qubit, 1],
                                        scaling[layer, qubit, 2] * inputs[qubit] + weights[layer, qubit, 2],
                                        wires=[(qubit + 1) % qubits])
            else:
                qml_CXYZ(scaling[layer, qubit, 0] * inputs[qubit] + weights[layer, qubit, 0],
                         scaling[layer, qubit, 1] * inputs[qubit] + weights[layer, qubit, 1],
                         scaling[layer, qubit, 2] * inputs[qubit] + weights[layer, qubit, 2],
                         qubit, (qubit + 1) % qubits)
        else:  # strongly entangled with parameterized single-qubit rotations
            if 'rot' == gates:
                qml.Rot(scaling[layer, qubit, 0] * inputs[qubit] + weights[layer, qubit, 0],
                        scaling[layer, qubit, 1] * inputs[qubit] + weights[layer, qubit, 1],
                        scaling[layer, qubit, 2] * inputs[qubit] + weights[layer, qubit, 2],
                        wires=qubit)
            elif 'u3' == gates:
                qml.U3(scaling[layer, qubit, 0] * inputs[qubit] + weights[layer, qubit, 0],
                       scaling[layer, qubit, 1] * inputs[qubit] + weights[layer, qubit, 1],
                       scaling[layer, qubit, 2] * inputs[qubit] + weights[layer, qubit, 2],
                       wires=qubit)
            else:
                qml_XYZ(scaling[layer, qubit, 0] * inputs[qubit] + weights[layer, qubit, 0],
                        scaling[layer, qubit, 1] * inputs[qubit] + weights[layer, qubit, 1],
                        scaling[layer, qubit, 2] * inputs[qubit] + weights[layer, qubit, 2],
                        qubit)

    @qml.qnode(dev, interface='torch', diff_method='backprop')
    def qnode(inputs, scaling, weights):

        # test if parameters are provided in right format:
        #   (batch, qubits) for inputs
        #   (layers, qubits, 3) for scaling and weights
        assert qubits == inputs.shape[-1]
        assert layers == scaling.shape[0] and qubits == scaling.shape[1] and 3 == scaling.shape[2]
        assert layers == weights.shape[0] and qubits == weights.shape[1] and 3 == weights.shape[2]

        # need to permute batch dimension to end
        if 2 == len(inputs.shape):
            inputs = torch.permute(inputs, (1, 0))

        # initial equal superposition
        for qubit in range(qubits):
            qml.Hadamard(wires=qubit)

        for layer in range(layers - 1):

            # dynamic (variational) part
            for qubit in range(qubits):
                variational(inputs, scaling, weights, qubit=qubit, layer=layer)

            # static (entanglement / hadamard) part
            for qubit in range(qubits):
                if 'cx' == ansatz:
                    qml.CNOT(wires=[qubit, (qubit + 1) % qubits])
                elif 'cz' == ansatz:
                    qml.CZ(wires=[qubit, (qubit + 1) % qubits])
                else:
                    qml.Hadamard(wires=qubit)

        # final dynamic (variational) part
        for qubit in range(qubits):
            variational(inputs, scaling, weights, qubit=qubit, layer=layers - 1)

        # return Pauli-Z expectation value of all individual qubits
        return [qml.expval(qml.PauliZ(wires=qubit)) for qubit in range(qubits)]

    weights_shape = {'scaling': (layers, qubits, 3), 'weights': (layers, qubits, 3)}
    return qml.qnn.TorchLayer(qnode, weights_shape)

    # drawer = qml.draw(qnode"
    # print(drawer(np.ones((qubits,)), np.ones((layers, qubits, 3)), np.ones((layers, qubits, 3))))