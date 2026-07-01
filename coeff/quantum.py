from itertools import combinations

import numpy as np
import torch


I = torch.eye(2, dtype=torch.complex128)
X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
H = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / np.sqrt(2)


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
    return torch.cos(angle / 2)[:, None, None] * eye- 1j * torch.sin(angle / 2)[:, None, None] * p


def states(x, theta):
    psi = torch.zeros((len(x), 16), dtype=torch.complex128)
    psi[:, 0] = 1
    pairs = list(combinations(range(4), 2))
    zz = torch.kron(Z, Z)

    for layer in range(3):
        for q in range(4):
            psi = apply(psi, H, (q,))
        for q in range(4):
            psi = apply(psi, rotation(theta[layer, q] * x[:, q], Z), (q,))
        for k, (q1, q2) in enumerate(pairs):
            angle = 2 * theta[layer, 4 + k] * x[:, q1] * x[:, q2]
            psi = apply(psi, rotation(angle, zz), (q1, q2))
    return psi


def rho(psi):
    return torch.einsum("bi,bj->ij", psi, psi.conj()) / len(psi)


def surrogate(psi, y):
    r0, r1 = rho(psi[y == 0]), rho(psi[y == 1])
    p0 = torch.real(torch.trace(r0 @ r0))
    p1 = torch.real(torch.trace(r1 @ r1))
    cross = torch.real(torch.trace(r0 @ r1))
    return 0.25 * (1 - p0) + 0.25 * (1 - p1) + 0.5 * cross


def optimize(args):
    run, x, y, init_std, epochs, lr, patience, seed = args
    torch.set_num_threads(1)
    torch.manual_seed(seed + run)
    x = torch.tensor(x, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.long)
    theta = torch.nn.Parameter(1 + init_std * torch.randn((3, 10), dtype=torch.float64))
    opt = torch.optim.Adam([theta], lr=lr)
    best_loss, best_theta, stale = 1e9, None, 0

    for _ in range(epochs):
        opt.zero_grad()
        loss = surrogate(states(x, theta), y)
        loss.backward()
        value = loss.item()
        if value < best_loss - 1e-11:
            best_loss = value
            best_theta = theta.detach().clone()
            stale = 0
        else:
            stale += 1
        opt.step()
        if stale == patience:
            break

    with torch.no_grad():
        theta.copy_(best_theta)

    theta.grad = None
    loss = surrogate(states(x, theta), y)
    loss.backward()
    return run, theta.detach().numpy(), loss.item(), theta.grad.norm().item()


def numpy_states(x, theta):
    with torch.no_grad():
        return states(torch.tensor(x, dtype=torch.float64), torch.tensor(theta, dtype=torch.float64)).numpy()


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
    paulis = {"X": np.array([[0, 1], [1, 0]], complex),
              "Y": np.array([[0, -1j], [1j, 0]], complex),
              "Z": np.array([[1, 0], [0, -1]], complex)}
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
    return f"{gate[0]}[w={'-'.join(map(str, gate[1]))},x={gate[2]}]"


def gamma_scores(psi, x, y, pool):
    r0, r1 = density(psi[y == 0]), density(psi[y == 1])
    delta = r1 - r0
    values, vectors = np.linalg.eigh(delta)
    sign = np.where(values > 1e-10, 1, np.where(values < -1e-10, -1, 0))
    witness = (vectors * sign) @ vectors.conj().T
    scores = []

    for _, _, feature, k in pool:
        h = x[:, feature]
        a0 = np.einsum("b,bi,bj->ij", h[y == 0], psi[y == 0], psi[y == 0].conj()) / np.sum(y == 0)
        a1 = np.einsum("b,bi,bj->ij", h[y == 1], psi[y == 1], psi[y == 1].conj()) / np.sum(y == 1)
        dot = -1j * (k @ a1 - a1 @ k - k @ a0 + a0 @ k)
        scores.append(np.real(0.5 * np.trace(witness @ dot)))
    return np.array(scores)


def perturb(psi, x, gate, t):
    h = x[:, gate[2]]
    kpsi = psi @ gate[3].T
    return np.cos(t * h)[:, None] * psi - 1j * np.sin(t * h)[:, None] * kpsi
