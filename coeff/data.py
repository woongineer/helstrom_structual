import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from ucimlrepo import fetch_ucirepo


def balanced_split(y, n_train, n_test, seed):
    rng = np.random.default_rng(seed)

    y = np.asarray(y)
    labels = np.unique(y)

    train_idx = []
    test_idx = []

    for label in labels:
        idx = np.where(y == label)[0]
        rng.shuffle(idx)

        required = n_train + n_test
        if len(idx) < required:
            raise ValueError(f"Class {label} has only {len(idx)} samples")

        train_idx.extend(idx[:n_train])
        test_idx.extend(idx[n_train:n_train + n_test])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return np.array(train_idx), np.array(test_idx)


def preprocess(x_train, x_test, reduction_sz=4):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    pca = PCA(n_components=reduction_sz, random_state=42)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)

    scaler = MinMaxScaler((0, 2 * np.pi))
    x_train = scaler.fit_transform(x_train)
    x_test = np.clip(scaler.transform(x_test), 0, 2 * np.pi)

    return x_train.astype(float), x_test.astype(float)


def load_mnist_raw():
    data = fetch_openml("mnist_784", version=1, as_frame=False)

    x = data.data.astype(float)
    y = data.target.astype(str)

    keep = np.isin(y, ["0", "1"])

    x = x[keep]
    y = (y[keep] == "1").astype(int)

    return x, y


def load_wine_raw(name):
    data = fetch_ucirepo(id=186)
    df = data.data.original.copy()

    if name == "wine_quality":
        df = df[df["quality"].isin([5, 6])].reset_index(drop=True)
        x = df.drop(columns=["quality", "color"]).to_numpy(dtype=float)
        y = (df["quality"].to_numpy() == 6).astype(int)

    elif name == "wine_color":
        x = df.drop(columns=["quality", "color"]).to_numpy(dtype=float)
        y = (df["color"].astype(str).to_numpy() == "white").astype(int)

    return x, y


def load_raw_data(name):
    if name == "mnist":
        return load_mnist_raw()

    if name in ["wine_quality", "wine_color"]:
        return load_wine_raw(name)


def load_data(name, n_train, n_test, seed, reduction_sz=4):
    x, y = load_raw_data(name)
    tr, te = balanced_split(y=y, n_train=int(n_train/2), n_test=int(n_test/2), seed=seed)
    x_train, x_test = preprocess(x_train=x[tr], x_test=x[te], reduction_sz=reduction_sz)

    y_train = y[tr]
    y_test = y[te]

    return x_train, x_test, y_train, y_test
