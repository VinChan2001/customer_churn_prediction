import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None

    def load_data(self, file_path):
        return pd.read_csv(file_path)

    def create_synthetic_data(self, n_samples=10000):
        np.random.seed(42)

        data = {
            'customer_id': range(1, n_samples + 1),
            'tenure': np.random.randint(1, 73, n_samples),
            'monthly_charges': np.random.normal(65, 20, n_samples),
            'total_charges': np.random.normal(2300, 1500, n_samples),
            'contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.25, 0.20]),
            'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
            'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.35, 0.45, 0.20]),
            'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.30, 0.50, 0.20]),
            'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.30, 0.50, 0.20]),
            'streaming_tv': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.40, 0.40, 0.20]),
            'paperless_billing': np.random.choice(['Yes', 'No'], n_samples, p=[0.60, 0.40]),
            'senior_citizen': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'partner': np.random.choice(['Yes', 'No'], n_samples, p=[0.50, 0.50]),
            'dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.30, 0.70])
        }

        df = pd.DataFrame(data)

        churn_prob = (
            0.1 +
            0.3 * (df['contract'] == 'Month-to-month') +
            0.2 * (df['payment_method'] == 'Electronic check') +
            0.15 * (df['tenure'] < 12) +
            0.1 * (df['monthly_charges'] > 80) +
            0.1 * (df['senior_citizen'] == 1) +
            0.05 * (df['online_security'] == 'No') +
            np.random.normal(0, 0.1, n_samples)
        )

        churn_prob = np.clip(churn_prob, 0, 1)
        df['churn'] = np.random.binomial(1, churn_prob, n_samples)

        return df

    def preprocess_features(self, df, is_training=True):
        df = df.copy()

        if 'customer_id' in df.columns:
            df = df.drop('customer_id', axis=1)

        target_col = 'churn'
        if target_col in df.columns:
            y = df[target_col]
            X = df.drop(target_col, axis=1)
        else:
            y = None
            X = df

        categorical_columns = X.select_dtypes(include=['object']).columns
        numerical_columns = X.select_dtypes(exclude=['object']).columns

        for col in categorical_columns:
            if is_training:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    X[col] = le.transform(X[col].astype(str))

        if is_training:
            X[numerical_columns] = self.scaler.fit_transform(X[numerical_columns])
            self.feature_names = X.columns.tolist()
        else:
            X[numerical_columns] = self.scaler.transform(X[numerical_columns])

        return X, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)