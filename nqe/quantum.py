from itertools import combinations

import numpy as np
import torch
from torch import nn


Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / np.sqrt(2)
PAIRS = [(0, 1), (1, 2), (2, 3), (3, 0)]


def network():
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
    ).double()


def apply(psi, u, wires):
    other = [q for q in range(4) if q not in wires]
    perm = [0] + [q + 1 for q in other + list(wires)]
    x = psi.reshape(len(psi), 2, 2, 2, 2).permute(perm)
    x = x.reshape(len(psi), 2 ** len(other), 2 ** len(wires))
    if u.ndim == 2:
        x = torch.einsum("bmd,ed->bme", x, u)
    else:
        x = torch.einsum("bmd,bed->bme", x, u)
    x = x.reshape(len(psi), 2, 2, 2, 2)
    inv = [0] * 5
    for new, old in enumerate(perm):
        inv[old] = new
    return x.permute(inv).reshape(len(psi), 16)


def rotation(angle, p):
    eye = torch.eye(len(p), dtype=torch.complex128)
    return (
        torch.cos(angle / 2)[:, None, None] * eye
        - 1j * torch.sin(angle / 2)[:, None, None] * p
    )


def states(z):
    psi = torch.zeros((len(z), 16), dtype=torch.complex128)
    psi[:, 0] = 1
    zz = torch.kron(Z, Z)

    for _ in range(3):
        for q in range(4):
            psi = apply(psi, H, (q,))
            psi = apply(psi, rotation(-2 * z[:, q], Z), (q,))
        for q1, q2 in PAIRS:
            angle = -2 * (np.pi - z[:, q1]) * (np.pi - z[:, q2])
            psi = apply(psi, rotation(angle, zz), (q1, q2))
    return psi


def fidelity(z1, z2):
    psi1, psi2 = states(z1), states(z2)
    overlap = torch.einsum("bi,bi->b", psi1.conj(), psi2)
    return torch.abs(overlap) ** 2


def pair_loss(model, x, y, batch_size):
    i = torch.randint(len(x), (batch_size,))
    j = torch.randint(len(x), (batch_size,))
    pred = fidelity(model(x[i]), model(x[j]))
    target = (y[i] == y[j]).double()
    return torch.mean((pred - target) ** 2)


def optimize(args):
    run, x, y, iterations, batch_size, lr, seed = args
    torch.set_num_threads(1)
    torch.manual_seed(seed + run)
    x = torch.tensor(x, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.long)
    model = network()
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    for _ in range(iterations):
        loss = pair_loss(model, x, y, batch_size)
        opt.zero_grad()
        loss.backward()
        opt.step()

    loss = pair_loss(model, x, y, batch_size)
    opt.zero_grad()
    loss.backward()
    grad = sum(p.grad.square().sum() for p in model.parameters()).sqrt()
    weights = [p.detach().numpy() for p in model.parameters()]
    return run, weights, loss.item(), grad.item()


def numpy_embedding(x, weights):
    model = network()
    with torch.no_grad():
        for p, value in zip(model.parameters(), weights):
            p.copy_(torch.tensor(value))
        z = model(torch.tensor(x, dtype=torch.float64))
        return z.numpy(), states(z).numpy()


def density(psi):
    return np.einsum("bi,bj->ij", psi, psi.conj()) / len(psi)


def trace_distance(psi, y):
    delta = density(psi[y == 1]) - density(psi[y == 0])
    return 0.5 * np.abs(np.linalg.eigvalsh(delta)).sum()


def full_pauli(p, wires):
    factors = [np.eye(2, dtype=complex) for _ in range(4)]
    for q in wires:
        factors[q] = p
    out = factors[0]
    for factor in factors[1:]:
        out = np.kron(out, factor)
    return out


def gate_pool():
    paulis = {
        "X": np.array([[0, 1], [1, 0]], complex),
        "Y": np.array([[0, -1j], [1j, 0]], complex),
        "Z": np.array([[1, 0], [0, -1]], complex),
    }
    pool = []
    for name, p in paulis.items():
        for q in range(4):
            for feature in range(4):
                pool.append(("R" + name, (q,), feature, full_pauli(p, (q,))))
        for wires in combinations(range(4), 2):
            for feature in range(4):
                pool.append(("R" + name + name, wires, feature, full_pauli(p, wires)))
    return pool


def label(gate):
    return f"{gate[0]}[w={'-'.join(map(str, gate[1]))},z={gate[2]}]"


def gamma_scores(psi, z, y, pool):
    r0, r1 = density(psi[y == 0]), density(psi[y == 1])
    delta = r1 - r0
    values, vectors = np.linalg.eigh(delta)
    sign = np.where(values > 1e-10, 1, np.where(values < -1e-10, -1, 0))
    witness = (vectors * sign) @ vectors.conj().T
    scores = []

    for _, _, feature, k in pool:
        h = z[:, feature]
        a0 = np.einsum(
            "b,bi,bj->ij", h[y == 0], psi[y == 0], psi[y == 0].conj()
        ) / np.sum(y == 0)
        a1 = np.einsum(
            "b,bi,bj->ij", h[y == 1], psi[y == 1], psi[y == 1].conj()
        ) / np.sum(y == 1)
        dot = -1j * (k @ a1 - a1 @ k - k @ a0 + a0 @ k)
        scores.append(np.real(0.5 * np.trace(witness @ dot)))
    return np.array(scores)


def perturb(psi, z, gate, t):
    h = z[:, gate[2]]
    kpsi = psi @ gate[3].T
    return np.cos(t * h)[:, None] * psi - 1j * np.sin(t * h)[:, None] * kpsi
