# Written by claude
"""
test_noise_sim.py -- checks noise_sim.py against an independent dense-matrix
reference (full 2^n x 2^n matrices built with Kronecker products). NumPy only.

    python test_noise_sim.py
"""
from functools import reduce

import numpy as np

from noise_sim import NoiseConfig, NoisyQuantumSim

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def rz(a):
    return np.diag([np.exp(-0.5j * a), np.exp(0.5j * a)])


def ry(a):
    return np.array([[np.cos(a / 2), -np.sin(a / 2)], [np.sin(a / 2), np.cos(a / 2)]],
                    dtype=complex)


def rot(phi, theta, omega):          # RZ(omega) RY(theta) RZ(phi), built from parts
    return rz(omega) @ ry(theta) @ rz(phi)


def op_on(n, m, w):
    mats = [I2] * n
    mats[w] = m
    return reduce(np.kron, mats)


def crot_dense(n, c, t, phi, theta, omega):
    R = rot(phi, theta, omega)
    N = 2 ** n
    U = np.zeros((N, N), dtype=complex)
    for b in range(N):
        bits = [(b >> (n - 1 - i)) & 1 for i in range(n)]
        if bits[c] == 0:
            U[b, b] += 1.0
        else:
            for tb in (0, 1):
                b2bits = list(bits)
                b2bits[t] = tb
                b2 = sum(bit << (n - 1 - i) for i, bit in enumerate(b2bits))
                U[b2, b] += R[tb, bits[t]]
    return U


def depol_dense(rho, n, w, p):
    if p <= 0:
        return rho
    out = (1 - p) * rho
    for P in (X, Y, Z):
        Pw = op_on(n, P, w)
        out = out + (p / 3) * (Pw @ rho @ Pw.conj().T)
    return out


def reference_expectations(x, scaling, weights, p1=0.0, p2=0.0, readout=0.0, offsets=None):
    L, n, _ = scaling.shape
    N = 2 ** n
    res = []
    for xi in x:
        ang = scaling * xi[None, :, None] + weights
        if offsets is not None:
            ang = ang + offsets
        psi = np.zeros(N, dtype=complex)
        psi[0] = 1
        rho = np.outer(psi, psi.conj())
        for q in range(n):
            U = op_on(n, H, q)
            rho = U @ rho @ U.conj().T
            rho = depol_dense(rho, n, q, p1)
        for layer in range(L):
            for q in range(n):
                t = (q + 1) % n
                U = crot_dense(n, q, t, *ang[layer, q])
                rho = U @ rho @ U.conj().T
                rho = depol_dense(rho, n, q, p2)
                rho = depol_dense(rho, n, t, p2)
            if layer < L - 1:
                for q in range(n):
                    U = op_on(n, H, q)
                    rho = U @ rho @ U.conj().T
                    rho = depol_dense(rho, n, q, p1)
        probs = np.real(np.diag(rho))
        bits = np.array([[(b >> (n - 1 - i)) & 1 for i in range(n)] for b in range(N)])
        zs = 1 - 2 * bits
        if readout > 0:
            # independent symmetric flips: <Z_i> scales by (1 - 2r)
            res.append((probs @ zs) * (1 - 2 * readout))
        else:
            res.append(probs @ zs)
    return np.array(res)


def make_case(n, L, B, seed):
    rng = np.random.default_rng(seed)
    scaling = rng.uniform(0, 2 * np.pi, (L, n, 3))
    weights = rng.uniform(0, 2 * np.pi, (L, n, 3))
    x = rng.uniform(0, 1, (B, n))
    return x, scaling, weights


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (f"   {detail}" if detail else ""))
    return ok


def main():
    all_ok = True
    for n, L in [(2, 3), (3, 2), (4, 3)]:
        x, s, w = make_case(n, L, B=4, seed=10 * n + L)
        sim = NoisyQuantumSim(s, w)

        sim.configure(NoiseConfig())
        d = np.max(np.abs(sim.expectations(x) - reference_expectations(x, s, w)))
        all_ok &= check(f"n={n} L={L} noiseless matches dense reference", d < 1e-10,
                        f"max|diff|={d:.2e}")

        sim.configure(NoiseConfig(depol_1q=0.01, depol_2q=0.05))
        d = np.max(np.abs(sim.expectations(x)
                          - reference_expectations(x, s, w, p1=0.01, p2=0.05)))
        all_ok &= check(f"n={n} L={L} depolarizing matches dense Kraus reference", d < 2e-5,
                        f"max|diff|={d:.2e}")

        sim.configure(NoiseConfig(readout_error=0.07))
        d = np.max(np.abs(sim.expectations(x)
                          - reference_expectations(x, s, w, readout=0.07)))
        all_ok &= check(f"n={n} L={L} readout error scales <Z> by (1-2r)", d < 1e-10,
                        f"max|diff|={d:.2e}")

        sim.configure(NoiseConfig(coherent_sigma=0.1, coherent_mode="static", seed=3))
        off = 0.1 * sim._static
        d = np.max(np.abs(sim.expectations(x)
                          - reference_expectations(x, s, w, offsets=off)))
        all_ok &= check(f"n={n} L={L} static coherent offsets match reference", d < 1e-10,
                        f"max|diff|={d:.2e}")

    # coherent noise: reproducibility, seed dependence, sigma=0 identity
    x, s, w = make_case(3, 2, 6, 99)
    sim = NoisyQuantumSim(s, w)
    clean = sim.expectations(x)
    sim.configure(NoiseConfig(coherent_sigma=0.0, seed=5))
    all_ok &= check("coherent sigma=0 equals clean", np.allclose(sim.expectations(x), clean))
    sim.configure(NoiseConfig(coherent_sigma=0.2, seed=5))
    a = sim.expectations(x)
    sim.configure(NoiseConfig(coherent_sigma=0.2, seed=5))
    b = sim.expectations(x)
    sim.configure(NoiseConfig(coherent_sigma=0.2, seed=6))
    c = sim.expectations(x)
    all_ok &= check("coherent static: same seed reproducible", np.allclose(a, b))
    all_ok &= check("coherent static: different seed differs", not np.allclose(a, c, atol=1e-6))
    all_ok &= check("coherent noise changes the output", not np.allclose(a, clean, atol=1e-6))
    sim.configure(NoiseConfig(coherent_sigma=0.2, coherent_mode="dynamic", seed=5))
    d1 = sim.expectations(np.repeat(x[:1], 4, axis=0))
    all_ok &= check("coherent dynamic: identical inputs get different noise",
                    not np.allclose(d1[0], d1[1], atol=1e-6))

    # shot noise: unbiased, variance (1 - e^2)/S per qubit
    x, s, w = make_case(3, 2, 1, 7)
    sim = NoisyQuantumSim(s, w)
    exact = sim.expectations(x)[0]
    S, B = 200, 6000
    sim.configure(NoiseConfig(shots=S, seed=1))
    est = sim.expectations(np.repeat(x, B, axis=0))
    mean_err = np.max(np.abs(est.mean(0) - exact))
    var_th = (1 - exact ** 2) / S
    var_ratio = est.var(0) / var_th
    all_ok &= check("shot noise is unbiased", mean_err < 4 * np.sqrt(var_th.max() / B),
                    f"max|mean-exact|={mean_err:.4f}")
    all_ok &= check("shot noise variance matches (1-e^2)/S",
                    np.all((var_ratio > 0.85) & (var_ratio < 1.15)),
                    f"ratios={np.round(var_ratio, 3)}")

    # larger system and chunking path
    x, s, w = make_case(6, 3, 130, 5)
    sim = NoisyQuantumSim(s, w, dm_chunk=64)
    sim.configure(NoiseConfig(depol_1q=0.0, depol_2q=0.02))
    z = sim.expectations(x)
    all_ok &= check("n=6 density-matrix run (3 chunks) is finite and in [-1, 1]",
                    np.all(np.isfinite(z)) and np.all(np.abs(z) <= 1 + 1e-6))
    sim.configure(NoiseConfig())
    zc = sim.expectations(x)
    sim.configure(NoiseConfig(depol_1q=0.02, depol_2q=0.2))
    zn = sim.expectations(x)
    all_ok &= check("strong depolarizing shrinks mean |<Z>|",
                    np.abs(zn).mean() < np.abs(zc).mean(),
                    f"{np.abs(zc).mean():.3f} -> {np.abs(zn).mean():.3f}")

    print("\nALL TESTS PASSED" if all_ok else "\nSOME TESTS FAILED")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()