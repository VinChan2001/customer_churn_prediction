import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import optuna
import joblib


class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.smote = SMOTE(random_state=42)

    def get_base_models(self):
        return {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'xgboost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'lightgbm': lgb.LGBMClassifier(random_state=42, verbose=-1),
            'svm': SVC(random_state=42, probability=True)
        }

    def train_models(self, X_train, y_train, use_smote=True):
        if use_smote:
            X_train_balanced, y_train_balanced = self.smote.fit_resample(X_train, y_train)
        else:
            X_train_balanced, y_train_balanced = X_train, y_train

        base_models = self.get_base_models()

        for name, model in base_models.items():
            print(f"Training {name}...")
            model.fit(X_train_balanced, y_train_balanced)
            self.models[name] = model

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        return metrics

    def evaluate_all_models(self, X_test, y_test):
        results = {}
        for name, model in self.models.items():
            results[name] = self.evaluate_model(model, X_test, y_test)
        return results

    def cross_validate_models(self, X, y, cv=5):
        cv_results = {}
        for name, model in self.models.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
            cv_results[name] = {
                'mean_cv_score': scores.mean(),
                'std_cv_score': scores.std()
            }
        return cv_results

    def optimize_xgboost(self, X_train, y_train, X_val, y_val, n_trials=50):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42,
                'eval_metric': 'logloss'
            }

            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, y_pred_proba)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        self.best_model = xgb.XGBClassifier(**best_params)
        self.best_model.fit(X_train, y_train)

        return best_params, study.best_value

    def get_feature_importance(self, model_name='random_forest'):
        if model_name in self.models:
            model = self.models[model_name]
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_
        return None

    def save_model(self, model_name, filepath):
        if model_name in self.models:
            joblib.dump(self.models[model_name], filepath)

    def load_model(self, filepath):
        return joblib.load(filepath)