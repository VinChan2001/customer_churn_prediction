import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go


class Visualizer:
    def __init__(self, style='seaborn-v0_8'):
        plt.style.use('default')
        sns.set_palette("husl")
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    def plot_data_distribution(self, df, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Data Distribution Analysis', fontsize=16)

        sns.countplot(data=df, x='churn', ax=axes[0,0])
        axes[0,0].set_title('Churn Distribution')

        sns.histplot(data=df, x='tenure', bins=30, ax=axes[0,1])
        axes[0,1].set_title('Tenure Distribution')

        sns.boxplot(data=df, x='churn', y='monthly_charges', ax=axes[1,0])
        axes[1,0].set_title('Monthly Charges by Churn')

        sns.countplot(data=df, x='contract', hue='churn', ax=axes[1,1])
        axes[1,1].set_title('Contract Type vs Churn')
        axes[1,1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_correlation_matrix(self, df, save_path=None):
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numerical_cols].corr()

        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_feature_importance(self, feature_names, importance_values, title='Feature Importance', save_path=None):
        feature_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=True)

        plt.figure(figsize=(10, 8))
        plt.barh(feature_df['feature'], feature_df['importance'])
        plt.xlabel('Importance')
        plt.title(title)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_model_comparison(self, results_dict, save_path=None):
        metrics_df = pd.DataFrame(results_dict).T
        metrics_df = metrics_df.drop('confusion_matrix', axis=1)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)

        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            metrics_df[metric].plot(kind='bar', ax=ax, color=self.colors)
            ax.set_title(f'{metric.capitalize()} Comparison')
            ax.set_ylabel(metric.capitalize())
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_confusion_matrices(self, results_dict, save_path=None):
        n_models = len(results_dict)
        cols = 3
        rows = (n_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)

        for i, (model_name, results) in enumerate(results_dict.items()):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]

            cm = results['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{model_name} Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')

        for i in range(n_models, rows * cols):
            row = i // cols
            col = i % cols
            fig.delaxes(axes[row, col] if rows > 1 else axes[col])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_roc_curves(self, models_dict, X_test, y_test, save_path=None):
        plt.figure(figsize=(10, 8))

        for name, model in models_dict.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(True)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def create_interactive_dashboard(self, df, save_path=None):
        fig = go.Figure()

        churn_counts = df['churn'].value_counts()
        fig.add_trace(go.Bar(
            x=['No Churn', 'Churn'],
            y=[churn_counts[0], churn_counts[1]],
            name='Churn Distribution'
        ))

        fig.update_layout(
            title='Customer Churn Analysis Dashboard',
            xaxis_title='Churn Status',
            yaxis_title='Count'
        )

        if save_path:
            fig.write_html(save_path)

        return fig