import multiprocessing as mp
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import load_data
from quantum import gamma_scores, gate_pool, label, numpy_states, optimize, perturb, trace_distance
from utils import get_logger


def run(name):
    out = Path(__file__).parent / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    xtr, xte, ytr, yte = load_data(name, N_TRAIN, N_TEST, DATA_SEED)

    logger.info(f"{name}: start {N_RUNS} circuits")

    with mp.get_context("fork").Pool(N_WORKERS) as pool:
        fitted = []
        for result in pool.imap_unordered(optimize, [(i, xtr, ytr, INIT_STD, ADAM_EPOCHS, ADAM_LR, PATIENCE, RUN_SEED) for i in range(N_RUNS)], ):
            fitted.append(result)
            logger.info(f"{name}: {len(fitted)}/{N_RUNS}, run={result[0]}, loss={result[2]:.8f}, grad={result[3]:.3e}")
    fitted.sort()

    pool = gate_pool()
    base, rows = [], []

    for count, (run_id, theta, loss, grad) in enumerate(fitted, 1):
        train_states = numpy_states(xtr, theta)
        test_states = numpy_states(xte, theta)
        t0_train = trace_distance(train_states, ytr)
        t0_test = trace_distance(test_states, yte)
        scores = gamma_scores(train_states, xtr, ytr, pool)

        best = np.argmax(np.abs(scores))
        best_gate = pool[best]
        best_gamma = scores[best]
        good_sign = 1 if best_gamma >= 0 else -1

        rng = np.random.default_rng(42000000 + run_id)
        random_idx = rng.integers(len(pool))
        random_gate = pool[random_idx]
        random_sign = rng.choice([-1, 1])

        base.append([run_id, loss, grad, t0_train, t0_test, best_gamma, label(best_gate), scores[random_idx],
                     label(random_gate), random_sign])

        variants = {"original": (None, 0, 0),
                    "good": (best_gate, good_sign, abs(best_gamma)),
                    "bad": (best_gate, -good_sign, -abs(best_gamma)),
                    "random": ( random_gate, random_sign, random_sign * scores[random_idx])}

        for epsilon in EPSILONS:
            for variant, (gate, direction, predicted) in variants.items():
                if gate is None:
                    train_value, test_value = t0_train, t0_test
                    gate_name = ""
                else:
                    train_value = trace_distance(perturb(train_states, xtr, gate, epsilon * direction), ytr)
                    test_value = trace_distance(perturb(test_states, xte, gate, epsilon * direction), yte)
                    gate_name = label(gate)

                rows.append([run_id, epsilon, variant, gate_name, direction, predicted, train_value, test_value,
                             train_value - t0_train, test_value - t0_test, (train_value - t0_train) / epsilon])
        logger.info(f"{name}: evaluated {count}/{N_RUNS}")

    base = pd.DataFrame(base, columns=["run", "surrogate_loss", "gradient_norm", "train_T", "test_T", "best_gamma",
                                        "best_gate", "random_gamma", "random_gate", "random_sign"])
    data = pd.DataFrame(rows, columns=["run", "epsilon", "variant", "gate", "direction", "predicted_slope", "train_T",
                                       "test_T", "train_delta", "test_delta", "observed_slope"])
    base.to_csv(out / "base.csv", index=False)
    data.to_csv(out / "results.csv", index=False)

    summary = (data.groupby(["epsilon", "variant"]).agg( train_T=("train_T", "mean"), train_std=("train_T", "std"),
                                                         test_T=("test_T", "mean"), test_std=("test_T", "std"),
                                                         train_delta=("train_delta", "mean"),
                                                         test_delta=("test_delta", "mean")).reset_index() )
    summary.to_csv(out / "summary.csv", index=False)
    plot(name, data, out)


def plot(name, data, out):
    colors = {"original": "black", "good": "C0", "bad": "C3", "random": "gray"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, split in zip(axes, ["train", "test"]):
        for variant in ["good", "original", "random", "bad"]:
            x = data[data.variant == variant].groupby("epsilon")[split + "_T"]
            ax.errorbar(x.mean().index, x.mean(),
                        yerr=1.96 * x.sem(), marker="o", capsize=3, label=variant, color=colors[variant])
        ax.set_xscale("log")
        ax.set_xlabel("epsilon")
        ax.set_ylabel(split + " trace distance")
        ax.grid(alpha=0.2)
    axes[1].legend()
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(out / "trace_distance.png", dpi=180)
    plt.close(fig)

    x = data[(data.epsilon == min(EPSILONS)) & (data.variant != "original") ]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for variant in ["good", "bad", "random"]:
        z = x[x.variant == variant]
        ax.scatter(z.predicted_slope, z.observed_slope, s=18, alpha=0.6, label=variant, color=colors[variant])
    low = min(x.predicted_slope.min(), x.observed_slope.min())
    high = max(x.predicted_slope.max(), x.observed_slope.max())
    ax.plot([low, high], [low, high], "k--")
    ax.set_xlabel("predicted slope")
    ax.set_ylabel("observed slope")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "slope.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    logger = get_logger(__name__)

    DATASETS = ["mnist"]  # ["mnist", "wine_quality", "wine_color"]
    N_RUNS = 3  # 100
    N_WORKERS = 1  # 16 or 50
    N_TRAIN = 50  # 500
    N_TEST = 10  # 100
    DATA_SEED = 42

    INIT_STD = 0.25
    ADAM_EPOCHS = 400
    ADAM_LR = 0.03
    PATIENCE = 80
    RUN_SEED = 14200

    EPSILONS = [0.001, 0.03]  # [0.001, 0.003, 0.01, 0.03, 0.1]

    for dataset in DATASETS:
        run(dataset)
