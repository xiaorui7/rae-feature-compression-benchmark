import numpy as np
from sklearn.decomposition import PCA

from .core import atomic_npz


class Full:
    name = "full_fp32"

    def fit(self, x, validation=None):
        self.dim = x.shape[1]
        return self

    def transform(self, x):
        return np.asarray(x, dtype=np.float32)

    def inverse_transform(self, x):
        return np.asarray(x, dtype=np.float32)

    def metadata(self):
        return {"method": self.name, "dim": self.dim, "dtype": "float32", "trainable_params": 0}

    def save(self, path):
        atomic_npz(path, method=np.array(self.name), dim=np.array(self.dim))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            obj = cls()
            obj.dim = int(f["dim"])
        return obj


class FP16(Full):
    name = "full_fp16_storage"

    def transform(self, x):
        return np.asarray(x, dtype=np.float16)

    def metadata(self):
        return {"method": self.name, "dim": self.dim, "dtype": "float16", "trainable_params": 0,
                "compute_dtype": "float32"}


class PCAMethod:
    name = "pca"

    def __init__(self, dim=128, seed=42):
        self.dim, self.seed = dim, seed

    def fit(self, x, validation=None):
        if self.dim > min(x.shape):
            raise ValueError("PCA dimension exceeds training matrix")
        self.model = PCA(n_components=self.dim, svd_solver="full", whiten=False)
        self.model.fit(np.asarray(x, dtype=np.float32))
        self.components = self.model.components_.astype(np.float32)
        self.mean = self.model.mean_.astype(np.float32)
        return self

    def transform(self, x):
        return (np.asarray(x, dtype=np.float32) - self.mean) @ self.components.T

    def inverse_transform(self, x):
        return np.asarray(x, dtype=np.float32) @ self.components + self.mean

    def metadata(self):
        return {"method": self.name, "dim": self.dim, "dtype": "float32", "solver": "full",
                "whiten": False, "seed": self.seed, "trainable_params": 0,
                "persistent_floats": int(self.components.size + self.mean.size)}

    def save(self, path):
        atomic_npz(path, method=np.array(self.name), dim=np.array(self.dim), seed=np.array(self.seed),
                   components=self.components, mean=self.mean)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            obj = cls(int(f["dim"]), int(f["seed"]))
            obj.components = f["components"].copy()
            obj.mean = f["mean"].copy()
        return obj


class Adapter:
    name = "linear_adapter"

    def __init__(self, dim=128, seed=42, epochs=50, batch=256, patience=5):
        self.dim, self.seed, self.epochs, self.batch, self.patience = dim, seed, epochs, batch, patience

    def fit(self, x, validation):
        import torch

        torch.manual_seed(self.seed)
        x = np.asarray(x, dtype=np.float32)
        validation = np.asarray(validation, dtype=np.float32)
        if self.dim >= x.shape[1]:
            raise ValueError("adapter must be a bottleneck")
        d = x.shape[1]
        down = torch.nn.Linear(d, self.dim)
        up = torch.nn.Linear(self.dim, d)
        model = torch.nn.Sequential(down, up)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
        train = torch.from_numpy(x)
        val = torch.from_numpy(validation)
        best, stale = float("inf"), 0
        rng = np.random.default_rng(self.seed)
        for epoch in range(self.epochs):
            model.train()
            for indices in np.array_split(rng.permutation(len(x)), max(1, (len(x) + self.batch - 1) // self.batch)):
                batch = train[indices]
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(model(batch), batch)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                score = float(torch.nn.functional.mse_loss(model(val), val))
            if score < best - 1e-7:
                best, stale = score, 0
                self.stopped_epoch = epoch + 1
                self.weights = (down.weight.detach().numpy().copy(), down.bias.detach().numpy().copy(),
                                up.weight.detach().numpy().copy(), up.bias.detach().numpy().copy())
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.validation_mse = best
        return self

    def transform(self, x):
        w, b, _, _ = self.weights
        return np.asarray(x, dtype=np.float32) @ w.T + b

    def inverse_transform(self, x):
        _, _, w, b = self.weights
        return np.asarray(x, dtype=np.float32) @ w.T + b

    def metadata(self):
        return {"method": self.name, "dim": self.dim, "seed": self.seed, "dtype": "float32",
                "trainable_params": int(sum(a.size for a in self.weights)),
                "stopped_epoch": self.stopped_epoch, "validation_mse": self.validation_mse}

    def save(self, path):
        atomic_npz(path, method=np.array(self.name), dim=np.array(self.dim), seed=np.array(self.seed),
                   stopped_epoch=np.array(self.stopped_epoch), validation_mse=np.array(self.validation_mse),
                   down_w=self.weights[0], down_b=self.weights[1], up_w=self.weights[2], up_b=self.weights[3])

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            obj = cls(int(f["dim"]), int(f["seed"]))
            obj.stopped_epoch, obj.validation_mse = int(f["stopped_epoch"]), float(f["validation_mse"])
            obj.weights = tuple(f[k].copy() for k in ("down_w", "down_b", "up_w", "up_b"))
        return obj


def load_method(path):
    with np.load(path, allow_pickle=False) as f:
        name = str(f["method"].item())
    return {"full_fp32": Full, "full_fp16_storage": FP16, "pca": PCAMethod,
            "linear_adapter": Adapter}[name].load(path)
