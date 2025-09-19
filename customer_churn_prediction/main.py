#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
from src.data_preprocessor import DataPreprocessor
from src.model_trainer import ModelTrainer
from src.visualizer import Visualizer


def main():
    print("=== Customer Churn Prediction ML Pipeline ===\n")

    # Initialize components
    preprocessor = DataPreprocessor()
    trainer = ModelTrainer()
    visualizer = Visualizer()

    # Create output directories
    os.makedirs('reports/figures', exist_ok=True)
    os.makedirs('models/saved', exist_ok=True)

    # 1. Data Generation and Loading
    print("1. Generating synthetic customer data...")
    df = preprocessor.create_synthetic_data(n_samples=10000)
    print(f"   ✓ Created dataset with {df.shape[0]} customers and {df.shape[1]} features")
    print(f"   ✓ Churn rate: {df['churn'].mean():.2%}")

    # 2. Data Preprocessing
    print("\n2. Preprocessing data...")
    X, y = preprocessor.preprocess_features(df, is_training=True)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    print(f"   ✓ Training set: {X_train.shape[0]} samples")
    print(f"   ✓ Test set: {X_test.shape[0]} samples")
    print(f"   ✓ Features: {X_train.shape[1]}")

    # 3. Model Training
    print("\n3. Training multiple models...")
    trainer.train_models(X_train, y_train, use_smote=True)
    print(f"   ✓ Trained {len(trainer.models)} models: {list(trainer.models.keys())}")

    # 4. Model Evaluation
    print("\n4. Evaluating models...")
    results = trainer.evaluate_all_models(X_test, y_test)

    # Create results summary
    results_summary = []
    for model_name, metrics in results.items():
        results_summary.append({
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1_score']:.4f}",
            'ROC-AUC': f"{metrics['roc_auc']:.4f}"
        })

    results_df = pd.DataFrame(results_summary)
    results_df = results_df.sort_values('ROC-AUC', ascending=False)

    print("\n   Model Performance Results:")
    print("   " + "="*70)
    for _, row in results_df.iterrows():
        print(f"   {row['Model']:<20} | ROC-AUC: {row['ROC-AUC']} | F1: {row['F1-Score']}")

    # 5. Best Model Analysis
    best_model_name = results_df.iloc[0]['Model']
    best_model = trainer.models[best_model_name]
    best_metrics = results[best_model_name]

    print(f"\n5. Best Model: {best_model_name}")
    print(f"   ✓ ROC-AUC Score: {best_metrics['roc_auc']:.4f}")
    print(f"   ✓ Precision: {best_metrics['precision']:.4f}")
    print(f"   ✓ Recall: {best_metrics['recall']:.4f}")

    # 6. Feature Importance
    print("\n6. Analyzing feature importance...")
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = best_model.feature_importances_
        feature_names = preprocessor.feature_names

        # Get top 5 features
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)

        print("   Top 5 Important Features:")
        for i, (_, row) in enumerate(importance_df.head().iterrows()):
            print(f"   {i+1}. {row['feature']}: {row['importance']:.4f}")

    # 7. Business Impact Analysis
    print("\n7. Business Impact Analysis...")
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    high_risk_threshold = 0.7
    high_risk_customers = (y_pred_proba >= high_risk_threshold).sum()

    # Calculate potential revenue impact
    avg_monthly_revenue = df['monthly_charges'].mean()
    annual_revenue_per_customer = avg_monthly_revenue * 12

    potential_churn_loss = y_test.sum() * annual_revenue_per_customer
    high_risk_value = high_risk_customers * annual_revenue_per_customer

    print(f"   ✓ High-risk customers identified: {high_risk_customers:,}")
    print(f"   ✓ Potential annual revenue at risk: ${high_risk_value:,.2f}")
    print(f"   ✓ Total potential churn loss: ${potential_churn_loss:,.2f}")
    print(f"   ✓ Model coverage of actual churners: {(high_risk_customers/y_test.sum()*100):.1f}%")

    # 8. Save Results
    print("\n8. Saving results...")

    # Save the best model
    trainer.save_model(best_model_name, f'models/saved/best_model_{best_model_name}.joblib')

    # Save results summary
    results_df.to_csv('reports/model_performance_summary.csv', index=False)

    # Save feature importance
    if hasattr(best_model, 'feature_importances_'):
        importance_df.to_csv('reports/feature_importance.csv', index=False)

    print(f"   ✓ Best model saved: models/saved/best_model_{best_model_name}.joblib")
    print("   ✓ Performance summary saved: reports/model_performance_summary.csv")
    print("   ✓ Feature importance saved: reports/feature_importance.csv")

    # 9. Generate Visualizations
    print("\n9. Generating visualizations...")
    try:
        # Data distribution plots
        visualizer.plot_data_distribution(df, 'reports/figures/data_distribution.png')

        # Model comparison
        visualizer.plot_model_comparison(results, 'reports/figures/model_comparison.png')

        # ROC curves
        visualizer.plot_roc_curves(trainer.models, X_test, y_test, 'reports/figures/roc_curves.png')

        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            visualizer.plot_feature_importance(
                feature_names, feature_importance,
                f'{best_model_name} Feature Importance',
                'reports/figures/feature_importance.png'
            )

        print("   ✓ Visualizations saved to reports/figures/")

    except Exception as e:
        print(f"   ⚠ Warning: Could not generate all visualizations: {e}")

    print("\n" + "="*70)
    print("✨ CUSTOMER CHURN PREDICTION PIPELINE COMPLETED SUCCESSFULLY! ✨")
    print("="*70)

    print(f"\n📊 KEY INSIGHTS:")
    print(f"• Best Model: {best_model_name} (ROC-AUC: {best_metrics['roc_auc']:.4f})")
    print(f"• Dataset: {df.shape[0]:,} customers with {df['churn'].mean():.1%} churn rate")
    print(f"• High-risk customers: {high_risk_customers:,} ({high_risk_customers/len(y_test):.1%} of test set)")
    print(f"• Potential revenue at risk: ${high_risk_value:,.2f} annually")

    print(f"\n📁 Generated Files:")
    print(f"• Jupyter notebook: notebooks/churn_analysis.ipynb")
    print(f"• Best model: models/saved/best_model_{best_model_name}.joblib")
    print(f"• Performance report: reports/model_performance_summary.csv")
    print(f"• Visualizations: reports/figures/")

    print(f"\n🚀 Next Steps:")
    print(f"• Run the Jupyter notebook for detailed analysis")
    print(f"• Deploy the model for real-time churn prediction")
    print(f"• Set up monitoring and retraining pipeline")
    print(f"• Implement targeted retention campaigns")


if __name__ == "__main__":
    main()