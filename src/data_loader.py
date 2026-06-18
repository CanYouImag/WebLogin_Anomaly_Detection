import pandas as pd
import numpy as np
import os
import glob


class DataLoader:
    def __init__(self, data_dir: str, processed_path: str):
        self.data_dir = data_dir
        self.processed_path = processed_path
        self.label_mapping = None

    def load_and_clean(self) -> pd.DataFrame:
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"在 {self.data_dir} 下未找到 CSV 文件")

        print(f"找到 {len(csv_files)} 个 CSV 文件，开始加载...")
        dfs = []
        for f in sorted(csv_files):
            print(f"  加载: {os.path.basename(f)}")
            df = pd.read_csv(f, low_memory=False)
            df.columns = df.columns.str.strip()
            df = df.loc[:, ~df.columns.duplicated()]
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)
        print(f"数据合并完成！总计 {len(df)} 条记录")

        label_col = [c for c in df.columns if c.lower() == 'label']
        if label_col:
            actual_label = label_col[0]
            df[actual_label] = df[actual_label].astype(str).str.strip()

            threshold = 100
            counts = df[actual_label].value_counts()
            rare = counts[counts < threshold].index.tolist()
            if rare:
                safe_rare = [r.encode('ascii', errors='replace').decode('ascii') for r in rare]
                print(f"稀有类别（< {threshold} 条）合并为 'Other_Attack': {safe_rare}")
                df[actual_label] = df[actual_label].replace(rare, 'Other_Attack')

            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df['label'] = le.fit_transform(df[actual_label])
            self.label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            safe_map = {k.encode('ascii', errors='replace').decode('ascii'): v
                        for k, v in sorted(self.label_mapping.items(), key=lambda x: x[1])}
            print(f"类别映射 ({len(safe_map)} 类):")
            for k, v in safe_map.items():
                print(f"  {v}: {k}")
            df.drop(columns=[actual_label], inplace=True)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        before = len(df)
        df.dropna(inplace=True)
        dropped = before - len(df)
        if dropped:
            print(f"已删除 {dropped} 条含缺失值/无穷值的记录")

        return df

    def save_processed(self, df: pd.DataFrame):
        os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)
        df.to_csv(self.processed_path, index=False)
        print(f"清洗后的数据已保存至: {self.processed_path}")

    def save_label_mapping(self, path):
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.label_mapping, path)
        print(f"类别映射已保存至: {path}")


if __name__ == "__main__":
    loader = DataLoader(
        data_dir="data/raw/MachineLearningCVE",
        processed_path="data/processed/cleaned_intrusion_data.csv"
    )
    data = loader.load_and_clean()
    loader.save_processed(data)
    loader.save_label_mapping("models/saved/label_mapping.pkl")
