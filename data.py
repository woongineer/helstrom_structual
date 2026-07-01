import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def take(y, n_train, n_test, seed):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for label in [0, 1]:
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        tr.extend(idx[:n_train])
        te.extend(idx[n_train:n_train + n_test])
    rng.shuffle(tr)
    rng.shuffle(te)
    return np.array(tr), np.array(te)


def preprocess(x_train, x_test):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    pca = PCA(4, random_state=42)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)

    scaler = MinMaxScaler((0, 2 * np.pi))
    x_train = scaler.fit_transform(x_train)
    x_test = np.clip(scaler.transform(x_test), 0, 2 * np.pi)
    return x_train.astype(float), x_test.astype(float)


def load_mnist(n_train, n_test, seed):
    data = fetch_openml("mnist_784", version=1, as_frame=False)
    x, y = data.data.astype(float), data.target.astype(str)
    keep = np.isin(y, ["0", "1"])
    original_idx = np.where(keep)[0]
    x, y = x[keep], (y[keep] == "1").astype(int)

    train_pool = original_idx < 60000
    test_pool = ~train_pool
    xtr, ytr = x[train_pool], y[train_pool]
    xte, yte = x[test_pool], y[test_pool]

    tr, _ = take(ytr, n_train, 0, seed)
    _, te = take(yte, 0, n_test, seed + 1)
    xtr, xte = preprocess(xtr[tr], xte[te])
    return xtr, xte, ytr[tr], yte[te]


def load_wine(name, n_train, n_test, seed):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
    red = pd.read_csv(url + "winequality-red.csv", sep=";")
    white = pd.read_csv(url + "winequality-white.csv", sep=";")
    red["color"] = 0
    white["color"] = 1
    data = pd.concat([red, white], ignore_index=True)

    if name == "wine_quality_56":
        data = data[data.quality.isin([5, 6])].reset_index(drop=True)
        y = (data.quality.to_numpy() == 6).astype(int)
    else:
        y = data.color.to_numpy().astype(int)

    x = data.drop(columns=["quality", "color"]).to_numpy(float)
    tr, te = take(y, n_train, n_test, seed)
    xtr, xte = preprocess(x[tr], x[te])
    return xtr, xte, y[tr], y[te]


def load_data(name, n_train, n_test, seed):
    if name == "mnist_01":
        return load_mnist(n_train, n_test, seed)
    return load_wine(name, n_train, n_test, seed)
