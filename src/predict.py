import os
import torch
import joblib
import pandas as pd
import numpy as np

from model import AnomalyDetectionMLP

_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_MODEL_DIR = os.path.join(_BASE, "models", "saved")


def _resolve(path):
    if not os.path.isabs(path):
        path = os.path.join(_MODEL_DIR, os.path.basename(path))
    return path


def predict_from_row(row_dict: dict,
                     stage1_path=None,
                     stage2_path=None,
                     scaler_path=None,
                     feature_names_path=None,
                     label_mapping_path=None,
                     stage2_mapping_path=None):
    stage1_path = _resolve(stage1_path or "stage1_binary.pth")
    stage2_path = _resolve(stage2_path or "stage2_attack.pth")
    scaler_path = _resolve(scaler_path or "scaler.pkl")
    feature_names_path = _resolve(feature_names_path or "feature_names.txt")
    label_mapping_path = _resolve(label_mapping_path or "label_mapping.pkl")
    stage2_mapping_path = _resolve(stage2_mapping_path or "stage2_label_mapping.pkl")

    with open(feature_names_path, 'r') as f:
        feature_order = [line.strip() for line in f if line.strip()]

    scaler = joblib.load(scaler_path)
    label_mapping = joblib.load(label_mapping_path)
    id_to_name = {v: k.encode('ascii', errors='replace').decode('ascii') for k, v in label_mapping.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features = pd.DataFrame([[float(row_dict.get(col, 0)) for col in feature_order]],
                            columns=feature_order)
    features = scaler.transform(features)
    input_tensor = torch.tensor(features, dtype=torch.float32).to(device)

    # Stage 1: Binary (BENIGN vs Attack)
    bin_model = AnomalyDetectionMLP(input_dim=len(feature_order), num_classes=2).to(device)
    bin_model.load_state_dict(torch.load(stage1_path, map_location=device))
    bin_model.eval()

    with torch.no_grad():
        bin_logits = bin_model(input_tensor)
        bin_probs = torch.softmax(bin_logits, dim=1).squeeze().cpu().numpy()

    attack_prob = float(bin_probs[1])

    if attack_prob < 0.5:
        print(f"预测: BENIGN (攻击概率: {attack_prob:.4f})")
        return 0, "BENIGN", attack_prob

    # Stage 2: Attack subtype
    att_model = AnomalyDetectionMLP(input_dim=len(feature_order), num_classes=12).to(device)
    att_model.load_state_dict(torch.load(stage2_path, map_location=device))
    att_model.eval()

    with torch.no_grad():
        att_logits = att_model(input_tensor)
        att_probs = torch.softmax(att_logits, dim=1).squeeze().cpu().numpy()

    if att_probs.ndim == 0:
        att_probs = att_probs.reshape(1)

    stage2_pred = int(np.argmax(att_probs))
    original_label = stage2_pred + 1

    try:
        stage2_mapping = joblib.load(stage2_mapping_path)
        label_name = stage2_mapping.get(stage2_pred, id_to_name.get(original_label, f"Attack_{stage2_pred}"))
    except FileNotFoundError:
        label_name = id_to_name.get(original_label, f"Attack_{stage2_pred}")

    print(f"预测: {label_name} (攻击概率: {attack_prob:.4f}, 置信度: {att_probs[stage2_pred]:.4f})")
    return original_label, label_name, attack_prob


def batch_predict(csv_path: str,
                  stage1_path=None,
                  stage2_path=None,
                  scaler_path=None,
                  feature_names_path=None,
                  label_mapping_path=None,
                  stage2_mapping_path=None,
                  output_path=None):
    stage1_path = _resolve(stage1_path or "stage1_binary.pth")
    stage2_path = _resolve(stage2_path or "stage2_attack.pth")
    scaler_path = _resolve(scaler_path or "scaler.pkl")
    feature_names_path = _resolve(feature_names_path or "feature_names.txt")
    label_mapping_path = _resolve(label_mapping_path or "label_mapping.pkl")
    stage2_mapping_path = _resolve(stage2_mapping_path or "stage2_label_mapping.pkl")

    with open(feature_names_path, 'r') as f:
        feature_order = [line.strip() for line in f if line.strip()]

    df = pd.read_csv(csv_path, low_memory=False)
    has_label = 'label' in df.columns

    X = df[feature_order].values.astype(np.float32)
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    label_mapping = joblib.load(label_mapping_path)
    id_to_name = {v: k.encode('ascii', errors='replace').decode('ascii') for k, v in label_mapping.items()}
    try:
        stage2_mapping = joblib.load(stage2_mapping_path)
    except FileNotFoundError:
        stage2_mapping = {}

    # Stage 1
    bin_model = AnomalyDetectionMLP(input_dim=len(feature_order), num_classes=2).to(device)
    bin_model.load_state_dict(torch.load(stage1_path, map_location=device))
    bin_model.eval()
    with torch.no_grad():
        bin_logits = bin_model(X_tensor)
        bin_probs = torch.softmax(bin_logits, dim=1)

    # Stage 2
    att_model = AnomalyDetectionMLP(input_dim=len(feature_order), num_classes=12).to(device)
    att_model.load_state_dict(torch.load(stage2_path, map_location=device))
    att_model.eval()
    with torch.no_grad():
        att_logits = att_model(X_tensor)
        att_probs = torch.softmax(att_logits, dim=1)

    is_attack = bin_probs[:, 1] >= 0.5
    att_preds = torch.argmax(att_probs, dim=1).cpu().numpy()
    final_preds = np.where(is_attack.cpu().numpy(), att_preds + 1, 0)

    names = [stage2_mapping.get(p - 1, id_to_name.get(p, "BENIGN"))
             if p > 0 else "BENIGN" for p in final_preds]

    result_df = df.copy()
    result_df['predicted_label_id'] = final_preds
    result_df['predicted_label_name'] = names

    if has_label:
        from sklearn.metrics import classification_report, f1_score, accuracy_score
        y_true = df['label'].values
        print(classification_report(y_true, final_preds, zero_division=0))
        macro_f1 = f1_score(y_true, final_preds, average='macro')
        acc = accuracy_score(y_true, final_preds)
        print(f"Macro F1: {macro_f1:.4f}, Accuracy: {acc:.4f}")

    if output_path:
        result_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

    return final_preds, names


if __name__ == "__main__":
    csv_path = os.path.join(_BASE, "data/processed/cleaned_intrusion_data.csv")
    df = pd.read_csv(csv_path, low_memory=False)

    print("=" * 50)
    print("Two-Stage MLP Inference Demo")
    print("=" * 50)

    # BENIGN sample
    print("\n--- BENIGN sample ---")
    predict_from_row(df.drop(columns=['label']).iloc[0].to_dict())

    # Attack sample
    attack_idx = df[df['label'] != 0].index[0]
    actual_label = df['label'].iloc[attack_idx]
    print(f"\n--- Attack sample (true label ID: {actual_label}) ---")
    predict_from_row(df.drop(columns=['label']).iloc[attack_idx].to_dict())
