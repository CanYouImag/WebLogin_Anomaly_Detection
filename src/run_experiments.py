import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from datasets import load_cicids2017, load_unsw_nb15
from split import split_arrays, to_numpy
from evaluate import binary_metrics, multiclass_metrics, save_per_class, save_results_table, class_names_for


RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "..", "results")


def _get_groups_cicids(X):
    names = list(X.columns)
    idx = {n: i for i, n in enumerate(names)}
    port = ["Destination Port"]
    durations = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets",
                 "Total Length of Fwd Packets", "Total Length of Bwd Packets",
                 "Fwd Packets/s", "Bwd Packets/s", "Down/Up Ratio"]
    pkt_len = [c for c in names if any(k in c for k in ["Packet Length", "Packet Size",
              "Fwd Packet Length", "Bwd Packet Length", "Fwd Avg Segment Size", "Bwd Avg Segment Size"])]
    rates = [c for c in names if any(k in c for k in ["Bytes/s", "Packets/s", "Bytes/Bulk", "Packets/Bulk", "Bulk Rate"])]
    iat = [c for c in names if "IAT" in c]
    flags = [c for c in names if "Flag" in c and "Header" not in c]
    header_subflow = [c for c in names if any(k in c for k in ["Header", "Subflow", "Init_Win", "act_data", "min_seg_size"])]
    active_idle = [c for c in names if any(k in c for k in ["Active", "Idle"])]
    groups = [port, durations, pkt_len, rates, iat, flags, header_subflow, active_idle]
    used = set()
    cleaned = []
    for g in groups:
        fresh = [c for c in g if c not in used]
        if fresh:
            used.update(fresh)
            cleaned.append([idx[c] for c in fresh])
    misc = [c for c in names if c not in used]
    if misc:
        cleaned.append([idx[c] for c in misc])
    return cleaned


def _prepare_datasets_for_model(X_train, X_val, X_test, scaler):
    return scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)


def _eval_block(model, y_true, Xte_s, class_names, is_binary, tag, seed_dir, seed=None):
    if hasattr(model, "predict_proba"):
        preds = model.predict(Xte_s)
        prob = model.predict_proba(Xte_s)
    elif isinstance(model, torch.nn.Module):
        model.eval()
        with torch.no_grad():
            logits = model(torch.Tensor(Xte_s))
            preds = logits.argmax(dim=1).numpy()
            prob = torch.softmax(logits, dim=1).numpy()
    else:
        preds = model.predict(Xte_s)
        prob = None
    if is_binary:
        prob_attack = prob[:, 1] if prob is not None else None
        return binary_metrics(y_true, preds, prob_attack)
    metric, pc = multiclass_metrics(y_true, preds, class_names, prob)
    save_per_class(pc, os.path.join(seed_dir, f"{tag}_per_class.csv"))
    return metric


def _stratified_sample(X, y, n, seed):
    classes = np.unique(y)
    rng = np.random.RandomState(seed)
    rows = []
    target_per_class = round(n / len(classes))
    for c in classes:
        c_idx = np.where(y == c)[0]
        if len(c_idx) == 0:
            continue
        k = target_per_class if len(c_idx) > target_per_class else len(c_idx)
        rows.append(rng.choice(c_idx, size=k, replace=len(c_idx) < target_per_class))
    rows = np.concatenate(rows)
    if len(rows) > n:
        rows = rng.choice(rows, size=n, replace=False)
    return np.sort(rows)


def run_single(seed, X, y, name, out_dir, epochs, patience, focal, smote,
               subsample=None, model_list=None):
    seed_dir = os.path.join(out_dir, f"seed{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    train_idx, val_idx, test_idx = split_arrays(X, y, "random", seed=seed,
                                                val_frac=0.1, test_frac=0.2)
    Xtr, Xv, Xte = to_numpy(X.iloc[train_idx]), to_numpy(X.iloc[val_idx]), to_numpy(X.iloc[test_idx])
    ytr, yv, yte = y[train_idx], y[val_idx], y[test_idx]

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xv_s, Xte_s = _prepare_datasets_for_model(Xtr, Xv, Xte, scaler)

    results = []
    y_true = yte
    is_binary = len(np.unique(y)) == 2
    class_names = class_names_for(name)
    n_class = len(np.unique(y))

    if subsample and subsample < Xtr_s.shape[0]:
        rows = _stratified_sample(Xtr_s, ytr, subsample, seed)
        Xtr_sub, ytr_sub = Xtr_s[rows], ytr[rows]
    else:
        Xtr_sub, ytr_sub = Xtr_s, ytr
    Xv_sub, yv_sub = Xv_s, yv

    def _models():
        requested = model_list or ["all"]

        if "linear" in requested or "all" in requested:
            from models2 import train_linear
            t0 = time.time()
            m = train_linear(Xtr_sub, ytr_sub, seed)[0]
            results.append({"model": "LogReg", "time": time.time() - t0,
                            **_eval_block(m, y_true, Xte_s, class_names, is_binary, "logreg", seed_dir)})

        if "rf" in requested or "all" in requested:
            from models2 import train_rf
            m, t = train_rf(Xtr_sub, ytr_sub, seed, n_estimators=200, max_depth=30)
            results.append({"model": "RF", "time": t,
                            **_eval_block(m, y_true, Xte_s, class_names, is_binary, "rf", seed_dir)})

        if "xgb" in requested or "all" in requested:
            from models2 import train_xgb
            m, t = train_xgb(Xtr_sub, ytr_sub, seed)
            results.append({"model": "XGB", "time": t,
                            **_eval_block(m, y_true, Xte_s, class_names, is_binary, "xgb", seed_dir)})

        if "mlp" in requested or "all" in requested:
            from models2 import train_plain_mlp
            t0 = time.time()
            m = train_plain_mlp(Xtr_sub, ytr_sub, Xv_sub, yv_sub, seed,
                                num_epochs=epochs, patience=patience, focal=focal, use_smote=smote)
            results.append({"model": "MLP", "time": time.time() - t0,
                            **_eval_block(m, y_true, Xte_s, class_names, is_binary, "mlp", seed_dir)})

        if "two_stage" in requested or "all" in requested:
            if is_binary:
                from models2 import train_plain_mlp, train_two_stage
                t0 = time.time()
                m = train_plain_mlp(Xtr_sub, ytr_sub, Xv_sub, yv_sub, seed,
                                    num_epochs=epochs, patience=patience, focal=focal, use_smote=smote)
                results.append({"model": "MLP-TwoStage(bi)", "time": time.time() - t0,
                                **_eval_block(m, y_true, Xte_s, class_names, is_binary, "two_stage", seed_dir)})
            else:
                from models2 import train_two_stage
                t0 = time.time()
                predict_fn, _, predict_proba = train_two_stage(
                    Xtr_sub, ytr_sub, Xv_sub, yv_sub, seed,
                    num_epochs=epochs, patience=patience, focal=focal
                )
                preds = predict_fn(Xte_s)
                metric, pc = multiclass_metrics(y_true, preds, class_names, y_prob=predict_proba(Xte_s))
                save_per_class(pc, os.path.join(seed_dir, "two_stage_per_class.csv"))
                results.append({"model": "MLP-TwoStage", "time": time.time() - t0, **metric})

        if "mvt2" in requested or "all" in requested:
            if is_binary:
                from models2 import train_mvt_mlp
                t0 = time.time()
                m = train_mvt_mlp(Xtr_sub, ytr_sub, Xv_sub, yv_sub, seed,
                                  [list(range(Xte_s.shape[1]))] if name != "cicids2017" else _get_groups_cicids(X),
                                  num_epochs=epochs, patience=patience, focal=focal, use_smote=smote)
                results.append({"model": "MVT-2MLP", "time": time.time() - t0,
                                **_eval_block(m, y_true, Xte_s, class_names, is_binary, "mvt2", seed_dir)})
            else:
                from models2 import train_two_stage_mvt
                t0 = time.time()
                predict_fn, _, predict_proba = train_two_stage_mvt(
                    Xtr_sub, ytr_sub, Xv_sub, yv_sub, seed, _get_groups_cicids(X),
                    num_epochs=epochs, patience=patience, focal=focal
                )
                preds = predict_fn(Xte_s)
                metric, pc = multiclass_metrics(y_true, preds, class_names, y_prob=predict_proba(Xte_s))
                save_per_class(pc, os.path.join(seed_dir, "mvt2_per_class.csv"))
                results.append({"model": "MVT-2MLP", "time": time.time() - t0, **metric})

        if "mvt" in requested or "all" in requested:
            from models2 import train_mvt_mlp, train_plain_mlp
            if name == "cicids2017":
                groups = _get_groups_cicids(X) if n_class > 2 else [list(range(Xtr_s.shape[1]))]
            else:
                n_feat = Xtr_s.shape[1]
                groups = [list(range(min(n_feat, n_feat // 2))), list(range(min(n_feat // 2, n_feat), n_feat))]
            t0 = time.time()
            if is_binary:
                m = train_plain_mlp(Xtr_sub, ytr_sub, Xv_sub, yv_sub, seed,
                                    num_epochs=epochs, patience=patience, focal=focal, use_smote=smote)
            else:
                m = train_mvt_mlp(Xtr_sub, ytr_sub, Xv_sub, yv_sub, seed, groups,
                                  num_epochs=epochs, patience=patience, focal=focal, use_smote=smote)
            results.append({"model": "MVT-MLP", "time": time.time() - t0,
                            **_eval_block(m, y_true, Xte_s, class_names, is_binary, "mvt", seed_dir)})

        if "softgroup" in requested or "all" in requested:
            from models2 import train_softgroup_mlp
            if name == "cicids2017":
                groups = _get_groups_cicids(X) if n_class > 2 else [list(range(Xtr_s.shape[1]))]
            else:
                n_feat = Xtr_s.shape[1]
                groups = [list(range(min(n_feat, n_feat // 2))), list(range(min(n_feat // 2, n_feat), n_feat))]
            t0 = time.time()
            if is_binary:
                m = train_plain_mlp(Xtr_sub, ytr_sub, Xv_sub, yv_sub, seed,
                                    num_epochs=epochs, patience=patience, focal=focal, use_smote=smote)
            else:
                m = train_softgroup_mlp(Xtr_sub, ytr_sub, Xv_sub, yv_sub, seed, groups,
                                        num_epochs=epochs, patience=patience, focal=focal, use_smote=smote)
            results.append({"model": "MVT-Soft", "time": time.time() - t0,
                            **_eval_block(m, y_true, Xte_s, class_names, is_binary, "softgroup", seed_dir)})

        if "softgroup2" in requested or "all" in requested:
            if is_binary:
                from models2 import train_plain_mlp
                t0 = time.time()
                m = train_plain_mlp(Xtr_sub, ytr_sub, Xv_sub, yv_sub, seed,
                                    num_epochs=epochs, patience=patience, focal=focal, use_smote=smote)
                results.append({"model": "MVT-2Soft(bi)", "time": time.time() - t0,
                                **_eval_block(m, y_true, Xte_s, class_names, is_binary, "softgroup2", seed_dir)})
            else:
                from models2 import train_two_stage_softgroup
                t0 = time.time()
                predict_fn, _, predict_proba = train_two_stage_softgroup(
                    Xtr_sub, ytr_sub, Xv_sub, yv_sub, seed, _get_groups_cicids(X),
                    num_epochs=epochs, patience=patience, focal=focal
                )
                preds = predict_fn(Xte_s)
                metric, pc = multiclass_metrics(y_true, preds, class_names, y_prob=predict_proba(Xte_s))
                save_per_class(pc, os.path.join(seed_dir, "softgroup2_per_class.csv"))
                results.append({"model": "MVT-2Soft", "time": time.time() - t0, **metric})

        if "joint2" in requested or "all" in requested:
            if is_binary:
                from models2 import train_plain_mlp
                t0 = time.time()
                m = train_plain_mlp(Xtr_sub, ytr_sub, Xv_sub, yv_sub, seed,
                                    num_epochs=epochs, patience=patience, focal=focal, use_smote=smote)
                results.append({"model": "MVT-Joint(bi)", "time": time.time() - t0,
                                **_eval_block(m, y_true, Xte_s, class_names, is_binary, "joint2", seed_dir)})
            else:
                from models2 import train_two_stage_joint
                t0 = time.time()
                predict_fn, _, predict_proba = train_two_stage_joint(
                    Xtr_sub, ytr_sub, Xv_sub, yv_sub, seed, _get_groups_cicids(X),
                    num_epochs=epochs, patience=patience, focal=focal
                )
                preds = predict_fn(Xte_s)
                metric, pc = multiclass_metrics(y_true, preds, class_names, y_prob=predict_proba(Xte_s))
                save_per_class(pc, os.path.join(seed_dir, "joint2_per_class.csv"))
                results.append({"model": "MVT-Joint", "time": time.time() - t0, **metric})

        if "jointgnn" in requested or "all" in requested:
            if is_binary:
                from models2 import train_plain_mlp
                t0 = time.time()
                m = train_plain_mlp(Xtr_sub, ytr_sub, Xv_sub, yv_sub, seed,
                                    num_epochs=epochs, patience=patience, focal=focal, use_smote=smote)
                results.append({"model": "MVT-GNN(bi)", "time": time.time() - t0,
                                **_eval_block(m, y_true, Xte_s, class_names, is_binary, "jointgnn", seed_dir)})
            else:
                from models2 import train_two_stage_joint, BiViewGNN
                t0 = time.time()
                predict_fn, _, predict_proba = train_two_stage_joint(
                    Xtr_sub, ytr_sub, Xv_sub, yv_sub, seed, _get_groups_cicids(X),
                    num_epochs=epochs, patience=patience, focal=focal,
                    model_factory=lambda: BiViewGNN(_get_groups_cicids(X),
                                                    len(np.unique(ytr_sub)))
                )
                preds = predict_fn(Xte_s)
                metric, pc = multiclass_metrics(y_true, preds, class_names, y_prob=predict_proba(Xte_s))
                save_per_class(pc, os.path.join(seed_dir, "jointgnn_per_class.csv"))
                results.append({"model": "MVT-GNN", "time": time.time() - t0, **metric})

        if "protopnn" in requested or "all" in requested:
            if is_binary:
                from models2 import train_plain_mlp
                t0 = time.time()
                m = train_plain_mlp(Xtr_sub, ytr_sub, Xv_sub, yv_sub, seed,
                                    num_epochs=epochs, patience=patience, focal=focal, use_smote=smote)
                results.append({"model": "MVT-Proto(bi)", "time": time.time() - t0,
                                **_eval_block(m, y_true, Xte_s, class_names, is_binary, "protopnn", seed_dir)})
            else:
                from models2 import train_two_stage_joint, BiProtoGNN
                t0 = time.time()
                predict_fn, _, predict_proba = train_two_stage_joint(
                    Xtr_sub, ytr_sub, Xv_sub, yv_sub, seed, _get_groups_cicids(X),
                    num_epochs=epochs, patience=patience, focal=focal, batch_size=4096,
                    model_factory=lambda: BiProtoGNN(_get_groups_cicids(X),
                                                     len(np.unique(ytr_sub)))
                )
                preds = predict_fn(Xte_s)
                metric, pc = multiclass_metrics(y_true, preds, class_names, y_prob=predict_proba(Xte_s))
                save_per_class(pc, os.path.join(seed_dir, "protopnn_per_class.csv"))
                results.append({"model": "MVT-Proto", "time": time.time() - t0, **metric})

    _models()
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(seed_dir, "metrics.csv"), index=False, float_format="%.4f")
    print(df.to_string(index=False))
    return df


def load_dataset(name, csv_path=None):
    if name == "cicids2017":
        return load_cicids2017(csv_path)
    if name == "unsw_nb15":
        return load_unsw_nb15(csv_path)
    raise ValueError(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["cicids2017", "unsw_nb15"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    p.add_argument("--models", nargs="+", default=["all"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--focal", action="store_true", default=True)
    p.add_argument("--no-focal", dest="focal", action="store_false")
    p.add_argument("--smote", action="store_true", default=True)
    p.add_argument("--no-smote", dest="smote", action="store_false")
    p.add_argument("--subsample", type=int, default=None)
    args = p.parse_args()
    torch.manual_seed(42)
    np.random.seed(42)

    all_frames = []
    for ds_name in args.datasets:
        print(f"\n{'='*60}\nDataset: {ds_name}\n{'='*60}")
        X, y, *rest = load_dataset(ds_name)
        n_class = len(np.unique(y))
        print(f"  samples={len(y)}, features={X.shape[1]}, classes={n_class}")
        ds_out = os.path.join(RESULTS_ROOT, ds_name)
        os.makedirs(ds_out, exist_ok=True)
        for seed in args.seeds:
            df = run_single(seed, X, y, ds_name, ds_out, args.epochs, args.patience,
                            args.focal, args.smote, args.subsample, args.models)
            df["seed"] = seed
            df["dataset"] = ds_name
            all_frames.append(df)

    summary = pd.concat(all_frames, ignore_index=True)
    summary_path = os.path.join(RESULTS_ROOT, "summary.csv")
    summary.to_csv(summary_path, index=False, float_format="%.4f")
    f1_col = "f1_macro" if "f1_macro" in summary.columns else ("f1" if "f1" in summary.columns else None)
    print(f"\n{'='*60}\nFull results: {summary_path}\n{'='*60}")
    if f1_col:
        grp = summary.groupby(["dataset", "model"], as_index=False)[f1_col].agg(["mean", "std"]).round(4)
        print(grp)


if __name__ == "__main__":
    main()