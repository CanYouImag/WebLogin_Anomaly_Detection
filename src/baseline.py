import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import xgboost as xgb

from feature_engineering import FeatureEngineer
from model import AnomalyDetectionMLP
from train import FocalLoss, compute_class_weights

SEEDS = [42, 123, 456]
NUM_CLASSES = 13
NUM_EPOCHS = 100


def train_mlp(X_train, y_train, X_val, y_val, X_test, y_test, seed, num_classes, use_smote=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_smote:
        min_count = min(np.bincount(y_train))
        k_neighbors = min(5, min_count - 1) if min_count > 1 else 1
        sampling_strategy = {c: max(5000, cnt) for c, cnt in enumerate(np.bincount(y_train)) if cnt < 5000}
        smote = SMOTE(random_state=seed, k_neighbors=k_neighbors, sampling_strategy=sampling_strategy)
        X_train_res, y_train_res = smote.fit_resample(X_train.values, y_train)
        print(f"    SMOTE: {len(X_train)} -> {len(X_train_res)} samples")
        train_features = X_train_res
        train_labels = y_train_res
    else:
        train_features = X_train.values
        train_labels = y_train

    train_dataset = TensorDataset(
        torch.tensor(train_features, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val.values, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)

    model = AnomalyDetectionMLP(input_dim=X_train.shape[1],
                                num_classes=num_classes).to(device)

    class_weights = compute_class_weights(train_labels, power=0.4).to(device)
    criterion = FocalLoss(gamma=2.0, alpha=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_val_loss = float('inf')
    best_model_state = None
    patience = 20
    patience_counter = 0

    t0 = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        val_loss_avg = val_loss / len(val_loader)
        scheduler.step()

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    train_time = time.time() - t0

    model.load_state_dict(best_model_state)
    model.eval()
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_test_tensor)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    return preds, train_time


def train_mlp_two_stage(X_train, y_train, X_val, y_val, X_test, y_test, seed, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    # Stage 1: Binary (BENIGN vs Attack)
    y_train_bin = (y_train > 0).astype(np.int64)
    y_val_bin = (y_val > 0).astype(np.int64)

    min_count = min(np.bincount(y_train_bin))
    k_neighbors = min(5, min_count - 1) if min_count > 1 else 1
    smote_bin = SMOTE(random_state=seed, k_neighbors=k_neighbors)
    X_bin_res, y_bin_res = smote_bin.fit_resample(X_train.values, y_train_bin)
    print(f"    Stage1 SMOTE: {len(X_train)} -> {len(X_bin_res)}")

    bin_dataset = TensorDataset(
        torch.tensor(X_bin_res, dtype=torch.float32),
        torch.tensor(y_bin_res, dtype=torch.long)
    )
    bin_loader = DataLoader(bin_dataset, batch_size=1024, shuffle=True)

    val_bin_dataset = TensorDataset(
        torch.tensor(X_val.values, dtype=torch.float32),
        torch.tensor(y_val_bin, dtype=torch.long)
    )
    val_bin_loader = DataLoader(val_bin_dataset, batch_size=1024, shuffle=False)

    bin_model = AnomalyDetectionMLP(input_dim=X_train.shape[1], num_classes=2).to(device)
    bin_weights = compute_class_weights(y_bin_res, power=0.4).to(device)
    criterion = FocalLoss(gamma=2.0, alpha=bin_weights)
    optimizer = optim.AdamW(bin_model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_loss = float('inf')
    best_bin_state = None
    patience = 20
    pc = 0
    for epoch in range(NUM_EPOCHS):
        bin_model.train()
        total_loss = 0
        for bx, by in bin_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(bin_model(bx), by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        bin_model.eval()
        val_loss = 0
        with torch.no_grad():
            for bx, by in val_bin_loader:
                bx, by = bx.to(device), by.to(device)
                val_loss += criterion(bin_model(bx), by).item()

        val_loss_avg = val_loss / len(val_bin_loader)
        scheduler.step()
        if val_loss_avg < best_loss:
            best_loss = val_loss_avg
            best_bin_state = bin_model.state_dict().copy()
            pc = 0
        else:
            pc += 1
        if pc >= patience:
            break

    bin_model.load_state_dict(best_bin_state)

    # Stage 2: Multi-class for attack subtypes only
    attack_mask = y_train > 0
    X_att = X_train.values[attack_mask]
    y_att = y_train[attack_mask] - 1  # shift 1-12 -> 0-11

    attack_mask_val = y_val > 0
    X_att_val = X_val.values[attack_mask_val]
    y_att_val = y_val[attack_mask_val] - 1

    num_attack_classes = num_classes - 1

    min_count_att = min(np.bincount(y_att))
    k_neighbors_att = min(5, min_count_att - 1) if min_count_att > 1 else 1
    sampling_strategy = {c: max(2000, cnt) for c, cnt in enumerate(np.bincount(y_att)) if cnt < 2000}
    smote_att = SMOTE(random_state=seed, k_neighbors=k_neighbors_att, sampling_strategy=sampling_strategy)
    X_att_res, y_att_res = smote_att.fit_resample(X_att, y_att)
    print(f"    Stage2 SMOTE: {len(X_att)} -> {len(X_att_res)}")

    att_dataset = TensorDataset(
        torch.tensor(X_att_res, dtype=torch.float32),
        torch.tensor(y_att_res, dtype=torch.long)
    )
    att_loader = DataLoader(att_dataset, batch_size=1024, shuffle=True)

    att_model = AnomalyDetectionMLP(input_dim=X_train.shape[1], num_classes=num_attack_classes).to(device)
    att_weights = compute_class_weights(y_att_res, power=0.4).to(device)
    criterion_att = FocalLoss(gamma=2.0, alpha=att_weights)
    optimizer_att = optim.AdamW(att_model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler_att = optim.lr_scheduler.CosineAnnealingLR(optimizer_att, T_max=NUM_EPOCHS)

    best_att_loss = float('inf')
    best_att_state = None
    pc_att = 0
    for epoch in range(NUM_EPOCHS):
        att_model.train()
        total_loss = 0
        for bx, by in att_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer_att.zero_grad()
            loss = criterion_att(att_model(bx), by)
            loss.backward()
            optimizer_att.step()
            total_loss += loss.item()

        att_model.eval()
        if len(X_att_val) > 0:
            val_att_tensor = torch.tensor(X_att_val, dtype=torch.float32).to(device)
            val_att_labels = torch.tensor(y_att_val, dtype=torch.long).to(device)
            with torch.no_grad():
                val_loss_att = criterion_att(att_model(val_att_tensor), val_att_labels).item()
            scheduler_att.step()
            if val_loss_att < best_att_loss:
                best_att_loss = val_loss_att
                best_att_state = att_model.state_dict().copy()
                pc_att = 0
            else:
                pc_att += 1
            if pc_att >= patience:
                break
        else:
            best_att_state = att_model.state_dict().copy()

    if best_att_state is not None:
        att_model.load_state_dict(best_att_state)

    train_time = time.time() - t0

    # Inference: combine both stages
    bin_model.eval()
    att_model.eval()
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    with torch.no_grad():
        bin_logits = bin_model(X_test_tensor)
        bin_probs = torch.softmax(bin_logits, dim=1)
        is_attack = bin_probs[:, 1] >= 0.5

        att_logits = att_model(X_test_tensor)
        att_preds = torch.argmax(att_logits, dim=1).cpu().numpy()

    preds = np.where(is_attack.cpu().numpy(), att_preds + 1, 0)
    return preds, train_time


def main():
    print("Loading data...")
    fe = FeatureEngineer("data/processed/cleaned_intrusion_data.csv")
    fe.load_data().scale_features()
    X, y = fe.get_features_and_labels()

    results = []

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Seed = {seed}")
        print(f"{'='*60}")

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=2 / 3, random_state=seed, stratify=y_temp
        )

        X_train_np = X_train.values.astype(np.float32)
        X_test_np = X_test.values.astype(np.float32)

        seed_results = {'seed': seed}

        # --- Random Forest ---
        print("  Training Random Forest...")
        t0 = time.time()
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=30, n_jobs=-1,
            random_state=seed, verbose=0
        )
        rf.fit(X_train_np, y_train)
        rf_time = time.time() - t0
        rf_preds = rf.predict(X_test_np)
        rf_f1 = f1_score(y_test, rf_preds, average='macro')
        rf_acc = accuracy_score(y_test, rf_preds)
        seed_results['RF'] = (rf_f1, rf_acc, rf_time)
        print(f"    Random Forest: Macro F1={rf_f1:.4f}, Acc={rf_acc:.4f}, Time={rf_time:.1f}s")

        # --- XGBoost ---
        print("  Training XGBoost...")
        t0 = time.time()
        xgb_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=12, learning_rate=0.1,
            n_jobs=-1, random_state=seed, verbosity=0,
            tree_method='hist'
        )
        xgb_model.fit(X_train_np, y_train)
        xgb_time = time.time() - t0
        xgb_preds = xgb_model.predict(X_test_np)
        xgb_f1 = f1_score(y_test, xgb_preds, average='macro')
        xgb_acc = accuracy_score(y_test, xgb_preds)
        seed_results['XGB'] = (xgb_f1, xgb_acc, xgb_time)
        print(f"    XGBoost:       Macro F1={xgb_f1:.4f}, Acc={xgb_acc:.4f}, Time={xgb_time:.1f}s")

        # --- MLP Two-Stage ---
        print("  Training MLP-TwoStage...")
        mlp_preds, mlp_time = train_mlp_two_stage(
            X_train, y_train, X_val, y_val, X_test, y_test, seed, NUM_CLASSES
        )
        mlp_f1 = f1_score(y_test, mlp_preds, average='macro')
        mlp_acc = accuracy_score(y_test, mlp_preds)
        seed_results['MLP'] = (mlp_f1, mlp_acc, mlp_time)
        print(f"    MLP-2Stage:   Macro F1={mlp_f1:.4f}, Acc={mlp_acc:.4f}, Time={mlp_time:.1f}s")

        results.append(seed_results)

    # --- Print summary table ---
    print("\n\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    header = f"{'Model':>12} | {'Seed':>5} | {'Macro F1':>9} | {'Accuracy':>9} | {'Time(s)':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        for model_name in ['RF', 'XGB', 'MLP']:
            f1, acc, t = r[model_name]
            print(f"{model_name:>12} | {r['seed']:>5d} | {f1:>9.4f} | {acc:>9.4f} | {t:>8.1f}")
    print("-" * len(header))

    avg_row = f"{'Avg':>12} | {'':>5} |"
    for model_name in ['RF', 'XGB', 'MLP']:
        f1s = [r[model_name][0] for r in results]
        accs = [r[model_name][1] for r in results]
        ts = [r[model_name][2] for r in results]
        avg_row += f" {np.mean(f1s):>9.4f} | {np.mean(accs):>9.4f} | {np.mean(ts):>8.1f}"
    print(avg_row)

    # --- Visualization ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {'MLP': '#00BCD4', 'RF': '#4CAF50', 'XGB': '#FF9800'}
    model_labels = {'MLP': 'MLP-2Stage', 'RF': 'Random Forest', 'XGB': 'XGBoost'}

    # Chart 1: Macro F1
    ax = axes[0]
    x = np.arange(len(SEEDS))
    width = 0.25
    models_order = ['RF', 'XGB', 'MLP']
    for i, model_name in enumerate(models_order):
        values = [r[model_name][0] for r in results]
        bars = ax.bar(x + i * width, values, width, label=model_labels[model_name],
                      color=colors[model_name], edgecolor='white')
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'Seed {s}' for s in SEEDS])
    ax.set_ylabel('Macro F1')
    ax.set_title('Macro F1 Score')
    ax.legend(fontsize=8)
    ax.set_ylim(0.5, 1.0)

    # Chart 2: Accuracy
    ax = axes[1]
    for i, model_name in enumerate(models_order):
        values = [r[model_name][1] for r in results]
        bars = ax.bar(x + i * width, values, width, label=model_labels[model_name],
                      color=colors[model_name], edgecolor='white')
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'Seed {s}' for s in SEEDS])
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy')
    ax.legend(fontsize=8)
    ax.set_ylim(0.90, 1.0)

    # Chart 3: Training Time
    ax = axes[2]
    for i, model_name in enumerate(models_order):
        values = [r[model_name][2] for r in results]
        bars = ax.bar(x + i * width, values, width, label=model_labels[model_name],
                      color=colors[model_name], edgecolor='white')
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f'{v:.0f}s', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'Seed {s}' for s in SEEDS])
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Training / Inference Time')
    ax.legend(fontsize=8)

    plt.suptitle('Baseline Comparison (CIC-IDS2017, 13 classes)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("models/saved/baseline_comparison.png", dpi=150, bbox_inches='tight')
    print(f"\nComparison chart saved to models/saved/baseline_comparison.png")


if __name__ == "__main__":
    main()
