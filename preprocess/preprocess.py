import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class Preprocess:
    def __init__(
        self,
        file_path: str = "student_habits_performance.csv",
        target_column: str = "exam_score",
        test_size: float = 0.1,
        random_state: int = 42,
    ):
        self.file_path = file_path
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state

    def preprocess_data(self):
        df = pd.read_csv(self.file_path)

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

        y = df[self.target_column].values
        X = df.drop(columns=[self.target_column])

        # one hot encoding categorical features
        categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
        X = pd.get_dummies(X, columns=categorical_columns, drop_first=False)

        # 90/10 split
        X_train, X_test, y_train, y_test = train_test_split(
            X.values,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        return X_train, X_test, y_train, y_test
