# Written by claude
""" 
noise_sim.py -- NumPy simulator of the Q-SpiRL quantum layer with toggleable noise.

Replicates the circuit in models/quantum_models.py for the default ansatz
('iqp') and gate set ('rot'):

    H on every qubit
    for each layer l:
        CRot(angles[l, q, :]) on wires [q, (q+1) % n]  for q = 0..n-1
        H on every qubit                                (all layers except the last)
    measure <Z> on every qubit

with angles[l, q, k] = scaling[l, q, k] * x[q] + weights[l, q, k].

Noise types (each one is off when its level is 0):
  shot      finite-shot estimate of <Z_i> from `shots` sampled bitstrings
  coherent  Gaussian over/under-rotation added to every CRot angle (std `coherent_sigma`,
            radians). mode 'static': one fixed draw per seed (a miscalibrated device);
            mode 'dynamic': a fresh draw for every sample and call.
  depol     single-qubit depolarizing channel, p = depol_1q after each H and
            p = depol_2q on EACH of the two wires after each CRot
            (two independent 1-qubit channels, an approximation of a 2-qubit one).
            Exact density-matrix simulation, so it is deterministic.
  readout   symmetric bit-flip probability `readout_error` on every measured bit.

Pure NumPy: no PennyLane or torch needed.
"""
import math
from dataclasses import dataclass

import numpy as np

NOISE_TYPES = ("shot", "coherent", "depol", "readout")

_H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0)


@dataclass(frozen=True)
class NoiseConfig:
    shots: int = 0                 # 0 = exact expectation values (no shot noise)
    coherent_sigma: float = 0.0    # radians; 0 = off
    coherent_mode: str = "static"  # "static" | "dynamic"
    depol_1q: float = 0.0          # 0 = off
    depol_2q: float = 0.0          # 0 = off
    readout_error: float = 0.0     # 0 = off
    seed: int = 0

    def any_noise(self):
        return (self.shots > 0 or self.coherent_sigma > 0 or self.depol_1q > 0
                or self.depol_2q > 0 or self.readout_error > 0)

    def is_stochastic(self):
        """Shot sampling and coherent noise depend on the seed; depolarizing and
        readout noise (exact simulation) do not."""
        return self.shots > 0 or self.coherent_sigma > 0

    def describe(self):
        parts = []
        if self.shots > 0:
            parts.append(f"shots={self.shots}")
        if self.coherent_sigma > 0:
            parts.append(f"coherent={self.coherent_sigma:g}({self.coherent_mode})")
        if self.depol_1q > 0 or self.depol_2q > 0:
            parts.append(f"depol_1q={self.depol_1q:g},depol_2q={self.depol_2q:g}")
        if self.readout_error > 0:
            parts.append(f"readout={self.readout_error:g}")
        return " + ".join(parts) if parts else "clean"


# --------------------------------------------------------------------------
# Tensor helpers
# --------------------------------------------------------------------------
def _apply_matrix(state, mat, axes):
    """Apply `mat` (2^k x 2^k, or batched (B, 2^k, 2^k)) to the k tensor axes `axes`
    of `state` (axis 0 is the batch axis). The first listed axis is the most
    significant bit of the matrix index."""
    k = len(axes)
    B = state.shape[0]
    mat = np.asarray(mat).astype(state.dtype, copy=False)
    moved = np.moveaxis(state, axes, list(range(1, k + 1)))
    shape = moved.shape
    flat = np.ascontiguousarray(moved).reshape(B, 2 ** k, -1)
    out = np.matmul(mat, flat).reshape(shape)
    return np.moveaxis(out, list(range(1, k + 1)), axes)


def _crot_matrices(phi, theta, omega):
    """Batched CRot(phi, theta, omega) matrices, control = first wire.
    Rot(phi, theta, omega) = RZ(omega) RY(theta) RZ(phi) (PennyLane convention)."""
    B = phi.shape[0]
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    U = np.zeros((B, 4, 4), dtype=np.complex128)
    U[:, 0, 0] = 1.0
    U[:, 1, 1] = 1.0
    U[:, 2, 2] = np.exp(-0.5j * (phi + omega)) * c
    U[:, 2, 3] = -np.exp(0.5j * (phi - omega)) * s
    U[:, 3, 2] = np.exp(-0.5j * (phi - omega)) * s
    U[:, 3, 3] = np.exp(0.5j * (phi + omega)) * c
    return U


def _depolarize(rho, wire, p, n):
    """rho -> (1-p) rho + p/3 (X rho X + Y rho Y + Z rho Z) on `wire`, written as
    (1-lam) rho + lam * (I/2 (x) tr_wire rho) with lam = 4p/3."""
    lam = 4.0 * p / 3.0
    r = np.moveaxis(rho, [1 + wire, 1 + n + wire], [-2, -1])
    tr = 0.5 * (r[..., 0, 0] + r[..., 1, 1])
    out = (1.0 - lam) * r
    out[..., 0, 0] += lam * tr
    out[..., 1, 1] += lam * tr
    return np.moveaxis(out, [-2, -1], [1 + wire, 1 + n + wire])


def _apply_readout(probs, r, n):
    if r <= 0:
        return probs
    C = np.array([[1.0 - r, r], [r, 1.0 - r]])
    p = probs.reshape((probs.shape[0],) + (2,) * n)
    for w in range(n):
        p = _apply_matrix(p, C, [1 + w])
    return p.reshape(probs.shape[0], -1)


# --------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------
class NoisyQuantumSim:
    def __init__(self, scaling, weights, ansatz="iqp", gates="rot", dm_chunk=64):
        if ansatz != "iqp" or gates != "rot":
            raise NotImplementedError("Only ansatz='iqp', gates='rot' (the defaults used "
                                      "by train.py) are implemented.")
        self.scaling = np.asarray(scaling, dtype=np.float64)
        self.weights = np.asarray(weights, dtype=np.float64)
        if self.scaling.shape != self.weights.shape or self.scaling.ndim != 3 \
                or self.scaling.shape[2] != 3:
            raise ValueError(f"scaling/weights must both be (layers, qubits, 3); got "
                             f"{self.scaling.shape} and {self.weights.shape}")
        self.L, self.n, _ = self.scaling.shape
        if self.n < 2:
            raise ValueError("need at least 2 qubits")
        self.dm_chunk = dm_chunk
        bits = (np.arange(2 ** self.n)[:, None] >> (self.n - 1 - np.arange(self.n))[None, :]) & 1
        self._zsign = 1.0 - 2.0 * bits          # (2^n, n): +1 for bit 0, -1 for bit 1
        self.configure(NoiseConfig())

    def configure(self, cfg):
        """Set the noise configuration (and reset the RNG streams from cfg.seed)."""
        self.cfg = cfg
        self._rng = np.random.default_rng([int(cfg.seed), 7])
        self._static = np.random.default_rng([int(cfg.seed), 11]).standard_normal(
            self.scaling.shape)

    # -- circuit -----------------------------------------------------------
    def _run(self, angles, mixed):
        n, L, c = self.n, self.L, self.cfg
        B = angles.shape[0]
        if mixed:
            state = np.zeros((B,) + (2,) * (2 * n), dtype=np.complex64)
            state[(slice(None),) + (0,) * (2 * n)] = 1.0
        else:
            state = np.zeros((B,) + (2,) * n, dtype=np.complex128)
            state[(slice(None),) + (0,) * n] = 1.0

        def gate(st, U, wires):
            st = _apply_matrix(st, U, [1 + w for w in wires])
            if mixed:
                st = _apply_matrix(st, np.conj(U), [1 + n + w for w in wires])
            return st

        def hadamards(st):
            for q in range(n):
                st = gate(st, _H, [q])
                if mixed and c.depol_1q > 0:
                    st = _depolarize(st, q, c.depol_1q, n)
            return st

        state = hadamards(state)
        for layer in range(L):
            for q in range(n):
                t = (q + 1) % n
                U = _crot_matrices(angles[:, layer, q, 0], angles[:, layer, q, 1],
                                   angles[:, layer, q, 2])
                state = gate(state, U, [q, t])
                if mixed and c.depol_2q > 0:
                    state = _depolarize(state, q, c.depol_2q, n)
                    state = _depolarize(state, t, c.depol_2q, n)
            if layer < L - 1:
                state = hadamards(state)

        if mixed:
            flat = state.reshape(B, 2 ** n, 2 ** n)
            probs = np.real(np.einsum("bii->bi", flat))
        else:
            probs = np.abs(state.reshape(B, 2 ** n)) ** 2
        return probs.astype(np.float64)

    def probabilities(self, x):
        """Outcome probabilities (B, 2^n) after gates, coherent and depolarizing noise
        (before readout error and shot sampling)."""
        x = np.asarray(x, dtype=np.float64).reshape(-1, self.n)
        B = x.shape[0]
        c = self.cfg
        angles = self.scaling[None] * x[:, None, :, None] + self.weights[None]
        if c.coherent_sigma > 0:
            if c.coherent_mode == "static":
                angles = angles + (c.coherent_sigma * self._static)[None]
            elif c.coherent_mode == "dynamic":
                angles = angles + c.coherent_sigma * self._rng.standard_normal(angles.shape)
            else:
                raise ValueError(f"unknown coherent_mode {c.coherent_mode!r}")
        mixed = c.depol_1q > 0 or c.depol_2q > 0
        chunk = self.dm_chunk if mixed else B
        out = [self._run(angles[i:i + chunk], mixed) for i in range(0, B, chunk)]
        return np.concatenate(out, axis=0)

    def _sample(self, probs, shots):
        est = np.empty((probs.shape[0], self.n))
        for i in range(probs.shape[0]):
            p = np.clip(probs[i], 0.0, None)
            p = p / p.sum()
            counts = self._rng.multinomial(shots, p)
            est[i] = counts @ self._zsign / shots
        return est

    def expectations(self, x):
        """<Z_i> for every qubit, shape (B, n), under the current noise config."""
        probs = _apply_readout(self.probabilities(x), self.cfg.readout_error, self.n)
        if self.cfg.shots > 0:
            return self._sample(probs, self.cfg.shots)
        return probs @ self._zsign