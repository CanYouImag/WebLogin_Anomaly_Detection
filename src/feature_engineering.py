import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os


class FeatureEngineer:
    def __init__(self, processed_path: str):
        self.processed_path = processed_path
        self.df = pd.DataFrame()
        self.scaler = StandardScaler()

    def load_data(self):
        if not os.path.exists(self.processed_path):
            raise FileNotFoundError(f"清洗后的数据文件未找到: {self.processed_path}")
        print(f"正在加载清洗后的数据: {self.processed_path}")
        self.df = pd.read_csv(self.processed_path, low_memory=False)
        print(f"数据加载完成！总计样本数: {len(self.df)}")
        return self

    def scale_features(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != 'label']
        if numeric_cols:
            print(f"正在标准化 {len(numeric_cols)} 个数值特征...")
            self.df[numeric_cols] = self.scaler.fit_transform(self.df[numeric_cols])
        return self

    def get_features_and_labels(self, target_col: str = 'label'):
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col].astype(int).values
        return X, y


if __name__ == "__main__":
    fe = FeatureEngineer("data/processed/cleaned_intrusion_data.csv")
    fe.load_data().scale_features()
    X, y = fe.get_features_and_labels()
    print(f"特征矩阵形状: {X.shape}")
    classes, counts = np.unique(y, return_counts=True)
    print(f"类别分布: {dict(zip(classes, counts))}")
