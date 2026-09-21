import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")


class GroupedMLP(nn.Module):
    def __init__(self, groups, num_classes, width=64, hidden=3, dropout=0.1):
        super().__init__()
        self.groups = [list(g) for g in groups]
        self.group_modules = nn.ModuleList()
        for cols in self.groups:
            self.group_modules.append(self._make_stream(len(cols), width, hidden, dropout))
        self.head = nn.Sequential(
            nn.Linear(width * len(self.groups), width),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(width, num_classes),
        )

    @staticmethod
    def _make_stream(in_dim, width, hidden, dropout):
        layers = [nn.Linear(in_dim, width), nn.LeakyReLU(0.1)]
        for _ in range(hidden - 1):
            layers += [nn.Linear(width, width), nn.LeakyReLU(0.1), nn.Dropout(dropout)]
        return nn.Sequential(*layers)

    def forward(self, x):
        views = []
        for cols, m in zip(self.groups, self.group_modules):
            views.append(m(x[:, cols]))
        return self.head(torch.cat(views, dim=1))


class SoftGroupedMLP(nn.Module):
    def __init__(self, num_features, groups, num_classes, width=64, hidden=3, dropout=0.1,
                 init_margin=4.0):
        super().__init__()
        n_groups = len(groups)
        logits = torch.zeros(num_features, n_groups)
        for k, cols in enumerate(groups):
            logits[cols, k] += init_margin
        self.assign_logits = nn.Parameter(logits)
        self.streams = nn.ModuleList(
            [self._make_stream(num_features, width, hidden, dropout) for _ in range(n_groups)]
        )
        self.head = nn.Sequential(
            nn.Linear(width * n_groups, width),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(width, num_classes),
        )

    @staticmethod
    def _make_stream(in_dim, width, hidden, dropout):
        layers = [nn.Linear(in_dim, width), nn.LeakyReLU(0.1)]
        for _ in range(hidden - 1):
            layers += [nn.Linear(width, width), nn.LeakyReLU(0.1), nn.Dropout(dropout)]
        return nn.Sequential(*layers)

    def forward(self, x):
        w = torch.softmax(self.assign_logits, dim=1)
        views = [m(x * w[:, k]) for k, m in enumerate(self.streams)]
        return self.head(torch.cat(views, dim=1))


class JointTwoStage(nn.Module):
    def __init__(self, groups, num_classes, width=64, hidden=3, dropout=0.1):
        super().__init__()
        self.groups = [list(g) for g in groups]
        self.group_modules = nn.ModuleList(
            self._make_stream(len(cols), width, hidden, dropout) for cols in self.groups
        )
        self.fuse = nn.Sequential(
            nn.Linear(width * len(self.groups), width),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )
        self.head1 = nn.Linear(width, 2)
        self.head2 = nn.Linear(width, num_classes - 1)

    @staticmethod
    def _make_stream(in_dim, width, hidden, dropout):
        layers = [nn.Linear(in_dim, width), nn.LeakyReLU(0.1)]
        for _ in range(hidden - 1):
            layers += [nn.Linear(width, width), nn.LeakyReLU(0.1), nn.Dropout(dropout)]
        return nn.Sequential(*layers)

    def forward(self, x, cond=None):
        views = [m(x[:, cols]) for cols, m in zip(self.groups, self.group_modules)]
        h = self.fuse(torch.cat(views, dim=1))
        return h, self.head1(h), self.head2(h)


class BiViewGNN(nn.Module):
    def __init__(self, groups, num_classes, width=64, hidden=3, dropout=0.1,
                 init_margin=4.0, num_rounds=1):
        super().__init__()
        n_features = sum(len(list(g)) for g in groups)
        n_groups = len(groups)
        logits = torch.zeros(n_features, n_groups)
        for k, cols in enumerate(groups):
            logits[list(cols), k] += init_margin
        self.assign_logits = nn.Parameter(logits)
        self.streams = nn.ModuleList(
            self._make_stream(n_features, width, hidden, dropout) for _ in range(n_groups)
        )
        self.gnorm = nn.LayerNorm(width)
        self.score = nn.Linear(width, 1)
        self.m_gain = nn.Parameter(torch.zeros(1))
        self.fuse = nn.Sequential(
            nn.Linear(3 * width, width),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
        )
        self.head1 = nn.Linear(width, 2)
        self.head2 = nn.Linear(width, num_classes - 1)
        self.width = width
        self.n_groups = n_groups
        self.num_rounds = num_rounds

    @staticmethod
    def _make_stream(in_dim, width, hidden, dropout):
        layers = [nn.Linear(in_dim, width), nn.LeakyReLU(0.1)]
        for _ in range(hidden - 1):
            layers += [nn.Linear(width, width), nn.LeakyReLU(0.1), nn.Dropout(dropout)]
        return nn.Sequential(*layers)

    def forward(self, x, cond=None):
        bs = x.size(0)
        w = torch.softmax(self.assign_logits, dim=1)
        U = torch.stack([m(x * w[:, k]) for k, m in enumerate(self.streams)], dim=1)
        a = torch.softmax(torch.stack([self.score(U[:, k]) for k in range(self.n_groups)], dim=1).squeeze(-1), dim=1)
        eps = 1e-6
        r = (U * a.unsqueeze(2)).sum(0) / (a.sum(0)[:, None] + eps)
        r = self.gnorm(r)
        alpha = torch.softmax((U * r.unsqueeze(0)).sum(-1) / self.width ** 0.5, dim=1)
        v1 = (U * a.unsqueeze(2)).sum(1)
        v2 = (U * alpha.unsqueeze(2)).sum(1)
        m = (r.unsqueeze(0) * alpha.unsqueeze(2)).sum(1) * torch.sigmoid(self.m_gain)
        h = self.fuse(torch.cat([v1, v2, m], dim=1))
        return h, self.head1(h), self.head2(h)


class BiProtoGNN(BiViewGNN):
    def __init__(self, groups, num_classes, *args, **kwargs):
        super().__init__(groups, num_classes, *args, **kwargs)
        self.class_prior = nn.Parameter(torch.zeros(num_classes))
        self.bin_prior = nn.Parameter(torch.zeros(2))

    def _prototype(self, U, a, eye, n_cond, prior):
        eps = 1e-6
        cnt = torch.einsum("nk,nc->kc", a, eye)
        proto = torch.einsum("nk,nc,nkd->kcd", a, eye, U) / (cnt[:, :, None] + eps)
        proto = self.gnorm(proto.reshape(-1, self.width)).view(self.n_groups, n_cond, self.width)
        valid = cnt > 1
        sim = torch.einsum("nkd,kcd->nkc", U, proto) / self.width ** 0.5
        sim = sim + prior[None, None, :].expand(sim.size(0), self.n_groups, n_cond)
        sim = sim.masked_fill(~valid.unsqueeze(0), float("-inf"))
        alpha = torch.softmax(sim.reshape(sim.size(0), -1), dim=1).view_as(sim)
        md = torch.einsum("nkc,kcd->nd", alpha, proto)
        return md, alpha

    def forward(self, x, cond=None):
        w = torch.softmax(self.assign_logits, dim=1)
        U = torch.stack([m(x * w[:, k]) for k, m in enumerate(self.streams)], dim=1)
        a = torch.softmax(torch.stack([self.score(U[:, k]) for k in range(self.n_groups)], dim=1).squeeze(-1), dim=1)
        eps = 1e-6
        r0 = (U * a.unsqueeze(2)).sum(0) / (a.sum(0)[:, None] + eps)
        r0 = self.gnorm(r0)
        v1 = (U * a.unsqueeze(2)).sum(1)
        v2 = (U * torch.softmax((U * r0.unsqueeze(0)).sum(-1) / self.width ** 0.5, dim=1).unsqueeze(2)).sum(1)
        h0 = self.fuse(torch.cat([v1, v2, v2 * torch.sigmoid(self.m_gain)], dim=1))
        c1, c2 = self.head1(h0), self.head2(h0)
        n_class = c2.size(1) + 1
        device = c1.device
        if cond is not None:
            labels, kind = cond
            labels = labels.to(device).long()
            if kind == "bin":
                eye = torch.stack([1 - labels, labels], dim=1).float()
                n_cond, prior = 2, self.bin_prior
            elif kind == "att":
                eye = torch.zeros(labels.size(0), n_class, device=device)
                eye[torch.arange(labels.size(0), device=device), labels + 1] = 1.0
                n_cond, prior = n_class, self.class_prior
            else:
                eye = torch.eye(n_class, device=device)[labels]
                n_cond, prior = n_class, self.class_prior
        else:
            pseudo = torch.where(c1[:, 1] > c1[:, 0], c2.argmax(dim=1) + 1, torch.zeros_like(c2.argmax(dim=1)))
            eye = torch.eye(n_class, device=device)[pseudo]
            n_cond, prior = n_class, self.class_prior
        md, _ = self._prototype(U, a, eye, n_cond, prior)
        m = md * torch.sigmoid(self.m_gain)
        h = self.fuse(torch.cat([v1, v2, m], dim=1))
        return h, self.head1(h), self.head2(h)


class PlainMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=(256, 128, 64, 32), dropout=0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LeakyReLU(0.1), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def class_weights(y, power=0.4):
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)
    raw = n / (len(classes) * counts.astype(float))
    return torch.tensor(raw ** power, dtype=torch.float32)


def _loader(X, y, batch_size, shuffle, seed=None):
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
    else:
        g = None
    ds = TensorDataset(torch.from_numpy(np.asarray(X, dtype=np.float32)),
                       torch.from_numpy(np.asarray(y, dtype=np.longlong)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g)


def _fit_torch(model, loader, val_loader, num_epochs, patience, seed, label,
               focal=True, alpha=None):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
    criterion = FocalLoss(gamma=2.0, alpha=alpha) if focal else nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_state = None
    pc = 0
    for epoch in range(num_epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            opt.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                val_loss += criterion(model(bx), by).item()
        val_loss /= max(1, len(val_loader))
        sched.step()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pc = 0
        else:
            pc += 1
        if pc >= patience:
            break
    model.load_state_dict(best_state)
    model.to("cpu")
    return model


def train_plain_mlp(X_train, y_train, X_val, y_val, seed, num_epochs=100,
                    patience=20, hidden=None, focal=True, use_smote=True, batch_size=1024):
    n_class = len(np.unique(y_train))
    weights = class_weights(y_train) if focal else None
    model = PlainMLP(X_train.shape[1], n_class, hidden_dims=hidden or (256, 128, 64, 32))
    if use_smote:
        counts = np.bincount(y_train)
        k = min(5, counts.min() - 1) if counts.min() > 1 else 1
        Xr, yr = SMOTE(random_state=seed, k_neighbors=k).fit_resample(X_train, y_train)
    else:
        Xr, yr = X_train, y_train
    loader = _loader(Xr, yr, batch_size, True, seed=seed)
    val_loader = _loader(X_val, y_val, batch_size, False)
    return _fit_torch(model, loader, val_loader, num_epochs, patience, seed, "mlp",
                      focal=focal, alpha=weights)


def train_mvt_mlp(X_train, y_train, X_val, y_val, seed, group_plan, num_epochs=100,
                  patience=20, focal=True, use_smote=True, batch_size=1024):
    n_class = len(np.unique(y_train))
    weights = class_weights(y_train) if focal else None
    model = GroupedMLP(group_plan, n_class)
    if use_smote:
        counts = np.bincount(y_train)
        k = min(5, counts.min() - 1) if counts.min() > 1 else 1
        Xr, yr = SMOTE(random_state=seed, k_neighbors=k).fit_resample(X_train, y_train)
    else:
        Xr, yr = X_train, y_train
    loader = _loader(Xr, yr, batch_size, True, seed=seed)
    val_loader = _loader(X_val, y_val, batch_size, False)
    return _fit_torch(model, loader, val_loader, num_epochs, patience, seed, "mvt",
                      focal=focal, alpha=weights)


def train_softgroup_mlp(X_train, y_train, X_val, y_val, seed, group_plan, num_epochs=100,
                        patience=20, focal=True, use_smote=True, batch_size=1024):
    n_class = len(np.unique(y_train))
    weights = class_weights(y_train) if focal else None
    model = SoftGroupedMLP(X_train.shape[1], group_plan, n_class)
    if use_smote:
        counts = np.bincount(y_train)
        k = min(5, counts.min() - 1) if counts.min() > 1 else 1
        Xr, yr = SMOTE(random_state=seed, k_neighbors=k).fit_resample(X_train, y_train)
    else:
        Xr, yr = X_train, y_train
    loader = _loader(Xr, yr, batch_size, True, seed=seed)
    val_loader = _loader(X_val, y_val, batch_size, False)
    return _fit_torch(model, loader, val_loader, num_epochs, patience, seed, "softgroup",
                      focal=focal, alpha=weights)


def _smote_balanced(X, y, seed, target_each=2000):
    counts = np.bincount(y)
    k = min(5, counts.min() - 1) if counts.min() > 1 else 1
    strategy = {c: max(target_each, cnt) for c, cnt in enumerate(counts) if cnt < target_each}
    return SMOTE(random_state=seed, k_neighbors=k,
                 sampling_strategy=strategy).fit_resample(X, y)


def train_two_stage(X_train, y_train, X_val, y_val, seed, num_epochs=100,
                    patience=20, batch_size=1024, focal=True, target_each=2000):
    n_class = len(np.unique(y_train))
    n_att = n_class - 1

    ybin = (y_train > 0).astype(np.int64)
    counts_bin = np.bincount(ybin)
    k = min(5, counts_bin.min() - 1) if counts_bin.min() > 1 else 1
    Xb, yb = SMOTE(random_state=seed, k_neighbors=k).fit_resample(X_train, ybin)
    loader_b = _loader(Xb, yb, batch_size, True, seed=seed)
    att_weights = class_weights(ybin)
    model_b = PlainMLP(X_train.shape[1], 2, hidden_dims=(128, 64, 32))
    stage1 = _fit_torch(model_b, loader_b,
                        _loader(X_val, (y_val > 0).astype(np.int64), batch_size, False),
                        num_epochs, patience, seed, "stage1", focal=focal, alpha=att_weights)

    m = y_train > 0
    Xa, ya = X_train[m], y_train[m] - 1
    if len(np.unique(ya)) < n_att:
        missing = set(range(n_att)) - set(np.unique(ya))
        raise ValueError(f"attack subtypes missing in train: {sorted(missing)}")
    Xa_r, ya_r = _smote_balanced(Xa, ya, seed, target_each=target_each)
    loader_a = _loader(Xa_r, ya_r, batch_size, True, seed=seed)
    m_val = y_val > 0
    Xav, yav = X_val[m_val], y_val[m_val] - 1
    if len(Xav) == 0:
        val_loader_a = loader_a
    else:
        val_loader_a = _loader(Xav, yav, batch_size, False)
    att_weights2 = class_weights(ya)
    model_a = PlainMLP(X_train.shape[1], n_att, hidden_dims=(256, 128, 64, 32))
    stage2 = _fit_torch(model_a, loader_a, val_loader_a, num_epochs, patience, seed, "stage2",
                        focal=focal, alpha=att_weights2)

    def predict(X):
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
        stage1.eval()
        stage2.eval()
        with torch.no_grad():
            p1 = torch.softmax(stage1(Xt), dim=1)[:, 1].numpy()
            p2 = stage2(Xt).argmax(dim=1).numpy()
        return np.where(p1 >= 0.5, p2 + 1, 0)

    def predict_proba(X):
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
        stage1.eval()
        stage2.eval()
        with torch.no_grad():
            p1 = torch.softmax(stage1(Xt), dim=1)[:, 1].numpy()[:, None]
            p2 = torch.softmax(stage2(Xt), dim=1).numpy() * p1
        proba = np.zeros((p1.shape[0], n_class), dtype=np.float32)
        proba[:, 0] = 1.0 - p1[:, 0]
        proba[:, 1:] = p2
        return proba

    return predict, (stage1, stage2), predict_proba


def train_two_stage_mvt(X_train, y_train, X_val, y_val, seed, group_plan, num_epochs=100,
                        patience=20, batch_size=1024, focal=True, target_each=2000):
    n_class = len(np.unique(y_train))
    n_att = n_class - 1

    ybin = (y_train > 0).astype(np.int64)
    counts_bin = np.bincount(ybin)
    k = min(5, counts_bin.min() - 1) if counts_bin.min() > 1 else 1
    Xb, yb = SMOTE(random_state=seed, k_neighbors=k).fit_resample(X_train, ybin)
    loader_b = _loader(Xb, yb, batch_size, True, seed=seed)
    stage1 = _fit_torch(GroupedMLP(group_plan, 2),
                        loader_b,
                        _loader(X_val, (y_val > 0).astype(np.int64), batch_size, False),
                        num_epochs, patience, seed, "stage1",
                        focal=focal, alpha=class_weights(ybin))

    m = y_train > 0
    Xa, ya = X_train[m], y_train[m] - 1
    if len(np.unique(ya)) < n_att:
        missing = set(range(n_att)) - set(np.unique(ya))
        raise ValueError(f"attack subtypes missing in train: {sorted(missing)}")
    Xa_r, ya_r = _smote_balanced(Xa, ya, seed, target_each=target_each)
    loader_a = _loader(Xa_r, ya_r, batch_size, True, seed=seed)
    m_val = y_val > 0
    Xav, yav = X_val[m_val], y_val[m_val] - 1
    if len(Xav) == 0:
        val_loader_a = loader_a
    else:
        val_loader_a = _loader(Xav, yav, batch_size, False)
    stage2 = _fit_torch(GroupedMLP(group_plan, n_att),
                        loader_a, val_loader_a, num_epochs, patience, seed, "stage2",
                        focal=focal, alpha=class_weights(ya))

    def predict(X):
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
        stage1.eval()
        stage2.eval()
        with torch.no_grad():
            p1 = torch.softmax(stage1(Xt), dim=1)[:, 1].numpy()
            p2 = stage2(Xt).argmax(dim=1).numpy()
        return np.where(p1 >= 0.5, p2 + 1, 0)

    def predict_proba(X):
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
        stage1.eval()
        stage2.eval()
        with torch.no_grad():
            p1 = torch.softmax(stage1(Xt), dim=1)[:, 1].numpy()[:, None]
            p2 = torch.softmax(stage2(Xt), dim=1).numpy() * p1
        proba = np.zeros((p1.shape[0], n_class), dtype=np.float32)
        proba[:, 0] = 1.0 - p1[:, 0]
        proba[:, 1:] = p2
        return proba

    return predict, (stage1, stage2), predict_proba


def train_two_stage_softgroup(X_train, y_train, X_val, y_val, seed, group_plan, num_epochs=100,
                              patience=20, batch_size=1024, focal=True, target_each=2000):
    n_class = len(np.unique(y_train))
    n_att = n_class - 1

    ybin = (y_train > 0).astype(np.int64)
    counts_bin = np.bincount(ybin)
    k = min(5, counts_bin.min() - 1) if counts_bin.min() > 1 else 1
    Xb, yb = SMOTE(random_state=seed, k_neighbors=k).fit_resample(X_train, ybin)
    loader_b = _loader(Xb, yb, batch_size, True, seed=seed)
    stage1 = _fit_torch(SoftGroupedMLP(X_train.shape[1], group_plan, 2),
                        loader_b,
                        _loader(X_val, (y_val > 0).astype(np.int64), batch_size, False),
                        num_epochs, patience, seed, "stage1",
                        focal=focal, alpha=class_weights(ybin))

    m = y_train > 0
    Xa, ya = X_train[m], y_train[m] - 1
    if len(np.unique(ya)) < n_att:
        missing = set(range(n_att)) - set(np.unique(ya))
        raise ValueError(f"attack subtypes missing in train: {sorted(missing)}")
    Xa_r, ya_r = _smote_balanced(Xa, ya, seed, target_each=target_each)
    loader_a = _loader(Xa_r, ya_r, batch_size, True, seed=seed)
    m_val = y_val > 0
    Xav, yav = X_val[m_val], y_val[m_val] - 1
    if len(Xav) == 0:
        val_loader_a = loader_a
    else:
        val_loader_a = _loader(Xav, yav, batch_size, False)
    stage2 = _fit_torch(SoftGroupedMLP(X_train.shape[1], group_plan, n_att),
                        loader_a, val_loader_a, num_epochs, patience, seed, "stage2",
                        focal=focal, alpha=class_weights(ya))

    def predict(X):
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
        stage1.eval()
        stage2.eval()
        with torch.no_grad():
            p1 = torch.softmax(stage1(Xt), dim=1)[:, 1].numpy()
            p2 = stage2(Xt).argmax(dim=1).numpy()
        return np.where(p1 >= 0.5, p2 + 1, 0)

    def predict_proba(X):
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
        stage1.eval()
        stage2.eval()
        with torch.no_grad():
            p1 = torch.softmax(stage1(Xt), dim=1)[:, 1].numpy()[:, None]
            p2 = torch.softmax(stage2(Xt), dim=1).numpy() * p1
        proba = np.zeros((p1.shape[0], n_class), dtype=np.float32)
        proba[:, 0] = 1.0 - p1[:, 0]
        proba[:, 1:] = p2
        return proba

    return predict, (stage1, stage2), predict_proba


def train_two_stage_joint(X_train, y_train, X_val, y_val, seed, group_plan, num_epochs=100,
                          patience=20, batch_size=1024, focal=True, target_each=2000, lam=1.0,
                          model_factory=None):
    from itertools import cycle

    n_class = len(np.unique(y_train))
    n_att = n_class - 1

    ybin = (y_train > 0).astype(np.int64)
    counts_bin = np.bincount(ybin)
    k = min(5, counts_bin.min() - 1) if counts_bin.min() > 1 else 1
    Xb, yb = SMOTE(random_state=seed, k_neighbors=k).fit_resample(X_train, ybin)

    m = y_train > 0
    Xa, ya = X_train[m], y_train[m] - 1
    if len(np.unique(ya)) < n_att:
        missing = set(range(n_att)) - set(np.unique(ya))
        raise ValueError(f"attack subtypes missing in train: {sorted(missing)}")
    Xa_r, ya_r = _smote_balanced(Xa, ya, seed, target_each=target_each)

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_factory is None:
        model_factory = lambda: JointTwoStage(group_plan, n_class)
    model = model_factory().to(device)
    criterion1 = FocalLoss(gamma=2.0, alpha=class_weights(ybin))
    criterion2 = FocalLoss(gamma=2.0, alpha=class_weights(ya))
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)

    loader_a = _loader(Xb, yb, batch_size, True, seed=seed)
    loader_v = _loader(X_val, y_val, batch_size, False)
    m_val = np.where(y_val > 0)[0]

    best_loss = float("inf")
    best_state = None
    pc = 0
    for epoch in range(num_epochs):
        model.train()
        it_b = iter(cycle(_loader(Xa_r, ya_r, batch_size, True, seed=seed)))
        for bx, by in loader_a:
            xb2, yb2 = next(it_b)
            bx, by = bx.to(device), by.to(device)
            xb2, yb2 = xb2.to(device), yb2.to(device)
            opt.zero_grad()
            _, c1, _ = model(bx, (by, "bin"))
            loss1 = criterion1(c1, by)
            _, _, c2 = model(xb2, (yb2, "att"))
            loss2 = criterion2(c2, yb2)
            (loss1 + lam * loss2).backward()
            opt.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in loader_v:
                bx, by = bx.to(device), by.to(device)
                if bx.size(0) == 0:
                    continue
                _, c1, c2 = model(bx, (by, "full"))
                l1 = criterion1(c1, (by > 0).long())
                mask_att = by > 0
                if mask_att.any():
                    l2 = criterion2(c2[mask_att], by[mask_att] - 1)
                else:
                    l2 = torch.zeros_like(l1)
                val_loss += (l1 + lam * l2).item() * bx.size(0)
        val_loss /= max(1, len(m_val))
        sched.step()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pc = 0
        else:
            pc += 1
            if pc >= patience:
                break
    model.load_state_dict(best_state)
    model.to("cpu")

    def predict(X):
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
        model.eval()
        with torch.no_grad():
            _, c1, c2 = model(Xt)
            p1 = torch.softmax(c1, dim=1)[:, 1].numpy()
            p2 = c2.argmax(dim=1).numpy()
        return np.where(p1 >= 0.5, p2 + 1, 0)

    def predict_proba(X):
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
        model.eval()
        with torch.no_grad():
            c1c, c2c = model(Xt)[1], model(Xt)[2]
            c1s = torch.softmax(c1c, dim=1).numpy()
            c2s = torch.softmax(c2c, dim=1).numpy()
        p1 = c1s[:, 1][:, None]
        proba = np.zeros((p1.shape[0], n_class), dtype=np.float32)
        proba[:, 0] = c1s[:, 0]
        proba[:, 1:] = c2s * p1
        return proba

    return predict, model, predict_proba


def train_rf(X_train, y_train, seed, n_estimators=200, max_depth=30):
    t0 = time.time()
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                   n_jobs=-1, random_state=seed, verbose=0)
    model.fit(X_train, y_train)
    return model, time.time() - t0


def train_xgb(X_train, y_train, seed, n_estimators=200, max_depth=12, lr=0.1):
    import xgboost as xgb
    t0 = time.time()
    model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                              learning_rate=lr, n_jobs=-1, random_state=seed,
                              verbosity=0, tree_method="hist")
    model.fit(X_train, y_train)
    return model, time.time() - t0


def train_catboost(X_train, y_train, seed):
    from catboost import CatBoostClassifier
    t0 = time.time()
    model = CatBoostClassifier(iterations=200, depth=8, verbose=0, random_seed=seed)
    model.fit(X_train, y_train)
    return model, time.time() - t0


def train_linear(X_train, y_train, seed):
    from sklearn.linear_model import LogisticRegression
    t0 = time.time()
    log_solver = "lbfgs" if X_train.shape[0] < 1e5 else "saga"
    model = LogisticRegression(max_iter=200, n_jobs=-1, solver=log_solver, random_state=seed)
    model.fit(X_train, y_train)
    return model, time.time() - t0