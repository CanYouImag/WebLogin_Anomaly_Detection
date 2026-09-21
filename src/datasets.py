import os
import numpy as np
import pandas as pd
import joblib


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")


def _cache_path(src_path, suffix="_cached.parquet"):
    return os.path.join(PROCESSED_DIR, os.path.splitext(os.path.basename(src_path))[0] + suffix)


def load_cicids2017(csv_path=None, use_cache=True, cache_dir=None):
    if csv_path is None:
        csv_path = os.path.join(PROCESSED_DIR, "cleaned_intrusion_data.csv")
    pk_path = _cache_path(csv_path)
    if use_cache and os.path.exists(pk_path):
        df = pd.read_parquet(pk_path)
    else:
        df = pd.read_csv(csv_path, low_memory=False)
        if use_cache:
            df.to_parquet(pk_path, index=False)
    label_col = "label"
    X = df.drop(columns=[label_col])
    y = df[label_col].astype(int).to_numpy()
    class_names = _cicids_class_names()
    return X, y, class_names


def _cicids_class_names():
    names = {
        0: "BENIGN",
        1: "Bot",
        2: "DDoS",
        3: "DoS GoldenEye",
        4: "DoS Hulk",
        5: "DoS Slowhttptest",
        6: "DoS slowloris",
        7: "FTP-Patator",
        8: "Other_Attack",
        9: "PortScan",
        10: "SSH-Patator",
        11: "Web Attack Brute Force",
        12: "Web Attack XSS",
    }
    return [names[i] for i in sorted(names)]


def load_unsw_nb15(csv_path=None, use_cache=True):
    if csv_path is None:
        candidates = [
            os.path.join(RAW_DIR, "UNSW-NB15", "UNSW_NB15_training-set.csv"),
            os.path.join(RAW_DIR, "UNSW-NB15", "UNSW-NB15-Balanced.csv"),
            os.path.join(RAW_DIR, "CSV Files", "Training and Testing Sets", "UNSW_NB15_training-set.csv"),
            os.path.join(RAW_DIR, "CSV Files", "Training and Testing Sets", "UNSW_NB15_testing-set.csv"),
        ]
        csv_path = next((c for c in candidates if os.path.exists(c)), None)
        if csv_path is None:
            raise FileNotFoundError(
                "UNSW-NB15 not found. Run scripts/prepare_unsw_nb15.py to download it."
            )
    pk_path = _cache_path(csv_path)
    if use_cache and os.path.exists(pk_path):
        df = pd.read_parquet(pk_path)
    else:
        df = pd.read_csv(csv_path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        if "id" in df.columns:
            df = df.drop(columns=["id"])
        if "attack_cat" in df.columns:
            df = df.drop(columns=["attack_cat"])
        if "label" not in df.columns:
            raise ValueError("UNSW-NB15 file must contain a 'label' column")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna().reset_index(drop=True)
        df = _encode_categorical(df)
        if use_cache:
            df.to_parquet(pk_path, index=False)
    X = df.drop(columns=["label"])
    y = df["label"].astype(int).to_numpy()
    return X, y, _unsw_class_names()


def _encode_categorical(df, cols=None):
    """One-hot encode nominal cols and label-encode high-cardinality proto."""
    import pandas.api.types as ptypes
    if cols is None:
        cols = ["proto", "service", "state"]
    for c in cols:
        if c not in df.columns:
            continue
        is_str = ptypes.is_string_dtype(df[c].dtype) or ptypes.is_object_dtype(df[c].dtype)
        if is_str:
            if c in ("proto", "proto_type"):
                df[c] = pd.Categorical(df[c]).codes.astype(np.int32)
            else:
                df = pd.concat([df, pd.get_dummies(df[c], prefix=c, dtype=np.int8)], axis=1)
                df = df.drop(columns=[c])
    return df


def _unsw_class_names():
    return ["Normal", "Attack"]


def load_nsl_kdd(base_path=None, use_cache=True):
    if base_path is None:
        base_path = os.path.join(RAW_DIR, "NSL-KDD")
    train_csv = os.path.join(base_path, "KDDTrain+.csv")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(
            "NSL-KDD not found. Put KDDTrain+.csv and KDDTest+.csv under data/raw/NSL-KDD/"
        )
    pk_train = _cache_path(train_csv)
    if use_cache and os.path.exists(pk_train):
        train_df = pd.read_parquet(pk_train)
        test_df = pd.read_parquet(_cache_path(os.path.join(base_path, "KDDTest+.csv")))
    else:
        cols = _nsl_kdd_columns()
        train_df = pd.read_csv(train_csv, header=None, names=cols, low_memory=False)
        test_df = pd.read_csv(os.path.join(base_path, "KDDTest+.csv"),
                              header=None, names=cols, low_memory=False)
        cat_cols = ["protocol_type", "service", "flag"]
        for df in (train_df, test_df):
            df["label"] = (df["label"].astype(str).str.endswith("normal")).astype(int)
            df.drop(columns=cat_cols, inplace=True)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
        if use_cache:
            train_df.to_parquet(pk_train, index=False)
            test_df.to_parquet(_cache_path(os.path.join(base_path, "KDDTest+.csv")), index=False)
    y_test = test_df["label"].astype(int).to_numpy()
    X_test = test_df.drop(columns=["label"])
    X = train_df.drop(columns=["label"])
    y = train_df["label"].astype(int).to_numpy()
    return X, y, X_test, y_test, _nsl_kdd_class_names()


def _nsl_kdd_columns():
    return (
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
        "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
        "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
        "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
        "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
    )


def _nsl_kdd_class_names():
    return ["Attack", "Normal"]


if __name__ == "__main__":
    X, y, names = load_cicids2017()
    print("CIC-IDS2017:", X.shape, "classes:", len(np.unique(y)))
    print(names)