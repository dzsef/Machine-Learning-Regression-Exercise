import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Preprocess:
    def __init__(
        self,
        file_path: str = "student_habits_performance.csv",
        target_column: str = "exam_score",
        test_size: float = 0.1,
        random_state: int = 42,
        scale = False
    ):
        self.file_path = file_path
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.scale = scale

    def preprocess_data(self):
        df = pd.read_csv(self.file_path)

        if self.file_path == "used_cars.csv":
            carname = df["CarName"].astype(str).apply(lambda x: x.strip())
            df["brand"] = carname.apply(lambda x: x.split(" ")[0] if x else "")
            df["model"] = carname.apply(lambda x: " ".join(x.split(" ")[1:]) if x else "")

        # drop rows with missing tartget
        df = df.dropna(subset=[self.target_column])

        nunique = df.nunique()

        constant_columns = [
            col for col in df.columns
            if nunique[col] == 1 and col != self.target_column
        ]

        # duplicated columns based on identical values
        duplicate_mask = df.T.duplicated()
        duplicate_columns = df.columns[duplicate_mask].tolist()
        if self.target_column in duplicate_columns:
            duplicate_columns.remove(self.target_column)

        # drop unique columns
        id_like_columns = []
        n_rows = len(df)
        for col in df.select_dtypes(include=["object", "category"]).columns:
            if nunique[col] > 0.9 * n_rows and col != self.target_column:
                id_like_columns.append(col)

        columns_to_drop = set(constant_columns + duplicate_columns + id_like_columns)

        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)

        # Imputing
        # numeric: median, categorical: Unknown
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

        for col in numeric_columns:
            if col == self.target_column:
                continue
            if df[col].isna().any():
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)

        for col in categorical_columns:
            if df[col].isna().any():
                df[col] = df[col].fillna("Unknown")

        # feature engineering for the used_cars dataset
        if self.file_path == "used_cars.csv":
            # power to weight ratio
            if "horsepower" in df.columns and "curbweight" in df.columns:
                df["power_to_weight_ratio"] = df["horsepower"] / df["curbweight"]

            # squared terms for original numeric features 
            for col in numeric_columns:
                if col == self.target_column:
                    continue
                if col in df.columns:
                    df[f"{col}_squared"] = df[col] ** 2

            # log transformed engine size
            if "enginesize" in df.columns:
                df["log_enginesize"] = np.log(df["enginesize"] + 1)

        y = df[self.target_column].values
        X = df.drop(columns=[self.target_column])

        # one hot encoding categorical features
        categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
        X = pd.get_dummies(X, columns=categorical_columns, drop_first=False)

        # keep feature names for plotting
        self.feature_names_ = X.columns.tolist()

        # 90/10 split
        X_train, X_test, y_train, y_test = train_test_split(
            X.values,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        
        if self.scale:
            scaler = StandardScaler()
            
            # Fit ONLY on training data to learn mean/std
            X_train = scaler.fit_transform(X_train)
            
            # Use the learned mean/std to transform test data
            X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test
