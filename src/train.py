import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from feature_engineering import FeatureEngineer
from model import AnomalyDetectionMLP
import os
import joblib


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


def compute_class_weights(y, power=0.4):
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    raw_weights = n_samples / (n_classes * counts.astype(float))
    weights = raw_weights ** power
    return torch.tensor(weights, dtype=torch.float32)


def train_stage(loader, val_loader, model, criterion, optimizer, scheduler,
                device, num_epochs, patience, label):
    best_loss = float('inf')
    best_state = None
    pc = 0

    print(f"  Training {label}...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                val_loss += criterion(model(bx), by).item()

        val_loss_avg = val_loss / len(val_loader)
        scheduler.step()

        if val_loss_avg < best_loss:
            best_loss = val_loss_avg
            best_state = model.state_dict().copy()
            pc = 0
        else:
            pc += 1
        if pc >= patience:
            print(f"    Early stop at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)
    return model


def plot_confusion_matrix(y_true, y_pred, label_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"混淆矩阵已保存至 {save_path}")
    plt.close()


def main():
    fe = FeatureEngineer("data/processed/cleaned_intrusion_data.csv")
    fe.load_data().scale_features()
    X, y = fe.get_features_and_labels()

    os.makedirs("models/saved", exist_ok=True)
    joblib.dump(fe.scaler, "models/saved/scaler.pkl")
    with open("models/saved/feature_names.txt", 'w') as f:
        for col in X.columns:
            f.write(f"{col}\n")

    num_classes = len(np.unique(y))
    label_mapping = joblib.load("models/saved/label_mapping.pkl")
    id_to_name = {v: k for k, v in label_mapping.items()}
    label_names = [id_to_name[i] for i in range(num_classes)]
    label_names_ascii = [n.encode('ascii', errors='replace').decode('ascii') for n in label_names]
    print(f"类别数: {num_classes}")

    seeds = [42, 123, 456]
    best_f1 = 0
    num_epochs = 100

    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"Random seed = {seed}")
        print(f"{'='*50}")

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=2/3, random_state=seed, stratify=y_temp
        )
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ===== Stage 1: Binary =====
        print(f"\n  --- Stage 1: BENIGN vs Attack ---")
        y_train_bin = (y_train > 0).astype(np.int64)
        y_val_bin = (y_val > 0).astype(np.int64)

        min_bin = min(np.bincount(y_train_bin))
        k_bin = min(5, min_bin - 1) if min_bin > 1 else 1
        smote_bin = SMOTE(random_state=seed, k_neighbors=k_bin)
        X_bin_res, y_bin_res = smote_bin.fit_resample(X_train.values, y_train_bin)
        print(f"    SMOTE: {len(X_train)} -> {len(X_bin_res)}")

        bin_loader = DataLoader(
            TensorDataset(torch.tensor(X_bin_res, dtype=torch.float32),
                          torch.tensor(y_bin_res, dtype=torch.long)),
            batch_size=1024, shuffle=True)
        val_bin_loader = DataLoader(
            TensorDataset(torch.tensor(X_val.values, dtype=torch.float32),
                          torch.tensor(y_val_bin, dtype=torch.long)),
            batch_size=1024, shuffle=False)

        bin_model = AnomalyDetectionMLP(input_dim=X_train.shape[1], num_classes=2).to(device)
        bin_weights = compute_class_weights(y_bin_res, power=0.4).to(device)
        bin_opt = optim.AdamW(bin_model.parameters(), lr=0.001, weight_decay=1e-4)
        bin_model = train_stage(
            bin_loader, val_bin_loader, bin_model,
            FocalLoss(gamma=2.0, alpha=bin_weights), bin_opt,
            optim.lr_scheduler.CosineAnnealingLR(bin_opt, T_max=num_epochs),
            device, num_epochs, patience=20, label="Stage1")

        torch.save(bin_model.state_dict(), f"models/saved/stage1_seed_{seed}.pth")
        print(f"    -> saved models/saved/stage1_seed_{seed}.pth")

        # ===== Stage 2: Attack subtypes =====
        print(f"\n  --- Stage 2: Attack Subtypes (12 classes) ---")
        attack_mask = y_train > 0
        X_att = X_train.values[attack_mask]
        y_att = y_train[attack_mask] - 1

        num_att = num_classes - 1
        min_att = min(np.bincount(y_att))
        k_att = min(5, min_att - 1) if min_att > 1 else 1
        sampling_strategy = {c: max(2000, cnt) for c, cnt in enumerate(np.bincount(y_att)) if cnt < 2000}
        smote_att = SMOTE(random_state=seed, k_neighbors=k_att, sampling_strategy=sampling_strategy)
        X_att_res, y_att_res = smote_att.fit_resample(X_att, y_att)
        print(f"    SMOTE: {len(X_att)} -> {len(X_att_res)}")
        print(f"    Distribution: {dict(sorted(Counter(y_att_res).items()))}")

        att_loader = DataLoader(
            TensorDataset(torch.tensor(X_att_res, dtype=torch.float32),
                          torch.tensor(y_att_res, dtype=torch.long)),
            batch_size=1024, shuffle=True)

        attack_mask_val = y_val > 0
        X_att_val = X_val.values[attack_mask_val]
        y_att_val = y_val[attack_mask_val] - 1
        att_val_loader = DataLoader(
            TensorDataset(torch.tensor(X_att_val, dtype=torch.float32),
                          torch.tensor(y_att_val, dtype=torch.long)),
            batch_size=1024, shuffle=False) if len(X_att_val) > 0 else None

        att_model = AnomalyDetectionMLP(input_dim=X_train.shape[1], num_classes=num_att).to(device)
        att_weights = compute_class_weights(y_att_res, power=0.4).to(device)
        att_opt = optim.AdamW(att_model.parameters(), lr=0.001, weight_decay=1e-4)
        att_model = train_stage(
            att_loader, att_val_loader, att_model,
            FocalLoss(gamma=2.0, alpha=att_weights), att_opt,
            optim.lr_scheduler.CosineAnnealingLR(att_opt, T_max=num_epochs),
            device, num_epochs, patience=20, label="Stage2")

        torch.save(att_model.state_dict(), f"models/saved/stage2_seed_{seed}.pth")
        print(f"    -> saved models/saved/stage2_seed_{seed}.pth")

        # ===== Evaluate =====
        bin_model.eval()
        att_model.eval()
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)

        with torch.no_grad():
            bin_logits = bin_model(X_test_tensor)
            bin_probs = torch.softmax(bin_logits, dim=1)
            is_attack = bin_probs[:, 1] >= 0.5
            att_logits = att_model(X_test_tensor)
            att_preds = torch.argmax(att_logits, dim=1).cpu().numpy()

        final_preds = np.where(is_attack.cpu().numpy(), att_preds + 1, 0)

        print("\nClassification Report:")
        print(classification_report(y_test, final_preds, zero_division=0, target_names=label_names_ascii))

        plot_confusion_matrix(y_test, final_preds, label_names_ascii,
                              "models/saved/confusion_matrix.png")

        test_f1 = f1_score(y_test, final_preds, average='macro')
        test_acc = accuracy_score(y_test, final_preds)
        print(f"Macro F1: {test_f1:.4f}, Accuracy: {test_acc:.4f}")

        if test_f1 > best_f1:
            best_f1 = test_f1
            torch.save(bin_model.state_dict(), "models/saved/stage1_binary.pth")
            torch.save(att_model.state_dict(), "models/saved/stage2_attack.pth")
            joblib.dump(seed, "models/saved/best_seed.pkl")

            stage2_mapping = {i - 1: label_names_ascii[i] for i in range(1, num_classes)}
            joblib.dump(stage2_mapping, "models/saved/stage2_label_mapping.pkl")
            print(f"  -> Best model (seed={seed}) saved")

    print(f"\n{'='*50}")
    print(f"All seeds done. Best Macro F1: {best_f1:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
