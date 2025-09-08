#!/usr/bin/env python3
"""
Random Forest Model Training System
NZ Unemployment Forecasting System - Random Forest Implementation

Features:
- Random Forest Regressor for all regions
- Regional-specific model training and evaluation
- Compressed model files for efficient storage
- Memory-efficient training pipeline
- Production-ready model artifacts

Author: Data Science Team
Version: Production v2.2 Random Forest
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
from datetime import datetime
import joblib
from typing import Dict, List, Optional, Tuple, Any, Union
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# ML Libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class UnemploymentModelTrainer:
    """
    Random Forest model training system for unemployment forecasting.
    
    This class provides Random Forest machine learning model training capabilities
    for regional unemployment forecasting. Designed for production use
    in government forecasting applications with robust evaluation and persistence.
    
    Model: Random Forest Regressor
    Target Regions: Auckland, Wellington, Canterbury
    """
    
    def __init__(self, data_dir: str = "model_ready_data", models_dir: str = "models", config_file: str = "simple_config.json") -> None:
        self.data_dir = Path(data_dir).resolve()
        self.models_dir = Path(models_dir).resolve()
        self.models_dir.mkdir(exist_ok=True)
        self.config_file = config_file
        
        # Load configuration
        self.config = self.load_config(config_file)
        
        # Load target regions from config
        self.target_regions = self.config.get('regions', {}).get('unemployment_core', ['Auckland', 'Wellington', 'Canterbury'])
        
        # Target columns will be detected from actual data, not hardcoded
        self.target_columns = []
        self.trained_models = {}
        self.model_performance = {}
        self.feature_importance = {}
        
        print("Multi-Algorithm Model Training System Initialized")
        print(f"Target Regions: {', '.join(self.target_regions)}")
        print(f"Data Directory: {self.data_dir}")
        print(f"Models Directory: {self.models_dir}")

    def load_config(self, config_file):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"Configuration loaded from {config_file}")
            return config
        except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
            print(f"WARNING: Failed to load config from {config_file}: {e}")
            return {}
        except Exception as e:
            print(f"ERROR: Unexpected error loading config from {config_file}: {e}")
            return {}

    def detect_target_columns(self, df):
        """Detect target columns using configuration-driven approach"""
        import re
        
        # Get forecasting configuration
        forecasting_config = self.config.get('forecasting', {})
        target_config = forecasting_config.get('target_columns', {})
        
        pattern = target_config.get('pattern', '.*unemployment_rate.*')
        exclude_patterns = target_config.get('exclude_patterns', ['lag', 'ma', 'change'])
        priority_regions = target_config.get('priority_regions', ['Auckland', 'Wellington', 'Canterbury'])
        priority_demographics = target_config.get('priority_demographics', ['European', 'Maori', 'Asian'])
        
        print(f"Detecting target columns with pattern: {pattern}")
        
        # Compile regex pattern
        try:
            regex_pattern = re.compile(pattern)
        except re.error:
            regex_pattern = re.compile(r".*unemployment_rate$")
        
        # Find candidate columns
        candidate_columns = []
        for col in df.columns:
            # Match both general and ethnic unemployment rate patterns
            if ('unemployment_rate' in col.lower() or regex_pattern.match(col.lower())):
                exclude = False
                for exclude_pattern in exclude_patterns:
                    if exclude_pattern.lower() in col.lower():
                        exclude = True
                        break
                if not exclude:
                    candidate_columns.append(col)
        
        # Prioritize columns - include ALL ethnic unemployment rates for demographic comparison
        priority_columns = []
        for col in candidate_columns:
            # Include ALL ethnic unemployment rates (client wants demographic comparison)
            for demo in priority_demographics:
                if demo in col:
                    priority_columns.append(col)
                    break
        
        target_columns = priority_columns if priority_columns else candidate_columns[:10]
        
        print(f"Found {len(target_columns)} target columns: {target_columns}")
        return target_columns

    def extract_region_from_column(self, column_name):
        """Extract region name from column name dynamically"""
        # Handle different column naming patterns
        if '_unemployment_rate' in column_name:
            # Remove the unemployment_rate suffix and extract last part as region
            base = column_name.replace('_unemployment_rate', '')
            parts = base.split('_')
            # Last part is typically the region
            return parts[-1] if parts else column_name
        return column_name

    def load_datasets(self):
        """Load train, validation, and test datasets"""
        print("\nLoading Datasets...")
        
        try:
            self.train_data = pd.read_csv(self.data_dir / "train_data.csv")
            self.validation_data = pd.read_csv(self.data_dir / "validation_data.csv")
            self.test_data = pd.read_csv(self.data_dir / "test_data.csv")
            
            # Load feature summary for metadata
            with open(self.data_dir / "feature_summary.json", 'r') as f:
                self.feature_summary = json.load(f)
            
            print(f"Training Data: {len(self.train_data)} records")
            print(f"Validation Data: {len(self.validation_data)} records") 
            print(f"Test Data: {len(self.test_data)} records")
            print(f"Total Features: {self.feature_summary.get('total_features', 'Unknown')}")
            
            # Detect target columns from the actual data
            self.target_columns = self.detect_target_columns(self.train_data)
            
            if not self.target_columns:
                print("ERROR: No target columns found!")
                return False
            
            return True
            
        except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            print(f"ERROR loading datasets - file/parsing issue: {e}")
            return False
        except Exception as e:
            print(f"ERROR loading datasets - unexpected issue: {e}")
            return False

    def prepare_features(self, dataset, target_col):
        """Prepare features for modeling by removing target and unnecessary columns"""
        # Remove target columns and non-predictive features
        exclude_cols = self.target_columns + ['date', 'quarter', 'year']
        feature_cols = [col for col in dataset.columns if col not in exclude_cols]
        
        # Smart feature imputation - limit forward-filling to prevent artificial repetition
        X = dataset[feature_cols].copy()
        buo_keywords = ['Computer_usage', 'ICT_', 'Loss_', 'Reason_', 'security', 'Outcomes']
        
        for col in feature_cols:
            if any(keyword in col for keyword in buo_keywords):
                # BUO columns: limited forward fill (max 1 period) then fill with 0
                X[col] = X[col].ffill(limit=1).fillna(0)
            else:
                # Other columns: standard imputation
                X[col] = X[col].ffill().fillna(0)
        
        # FIXED: Handle NaN in target variable properly
        y = dataset[target_col].ffill().bfill().fillna(dataset[target_col].mean())
        
        return X, y, feature_cols

    # ARIMA training removed - using Random Forest only

    # LSTM methods removed - not in top 3 performers (avg MAE: much higher than top 3)

    def train_random_forest_models(self):
        """Train Random Forest models for each target region"""
        print("\nTraining Random Forest Models...")
        
        rf_models = {}
        rf_performance = {}
        
        for target_col in self.target_columns:
            region = self.extract_region_from_column(target_col)
            print(f"\nTraining Random Forest for {region}...")
            
            try:
                # Prepare data
                X_train, y_train, feature_cols = self.prepare_features(self.train_data, target_col)
                X_val, y_val, _ = self.prepare_features(self.validation_data, target_col)
                
                if y_train.isna().sum() > len(y_train) * 0.95:  # Skip if >95% missing
                    print(f"WARNING Too much missing data for {region} Random Forest model")
                    continue
                
                # Handle missing data - fill both training and validation data
                y_train_filled = y_train.ffill().bfill().fillna(y_train.mean())
                X_train_filled = X_train.ffill().bfill().fillna(X_train.mean())
                
                # Also fill validation data to prevent NaN issues in evaluation
                y_val_filled = y_val.ffill().bfill().fillna(y_train.mean())  # Use training mean if val is all NaN
                X_val_filled = X_val.ffill().bfill().fillna(X_train.mean())
                
                # Check if we still have valid data after filling
                if y_train_filled.isna().sum() > 0 or X_train_filled.isna().sum().sum() > 0:
                    print(f"WARNING Persistent NaN values in training data for {region}, attempting backup filling")
                    # Backup: use median and zero-fill
                    y_train_filled = y_train_filled.fillna(y_train.median()).fillna(0)
                    X_train_filled = X_train_filled.fillna(X_train_filled.median()).fillna(0)
                    y_val_filled = y_val_filled.fillna(y_train.median()).fillna(0)
                    X_val_filled = X_val_filled.fillna(X_train_filled.median()).fillna(0)
                
                # Ensemble of Random Forest models for better variation
                rf_ensemble = []
                rf_predictions = []
                
                for seed in [None, 42, 123, 456]:  # Multiple models with different seeds
                    rf = RandomForestRegressor(
                        n_estimators=200,           # More trees for better predictions
                        max_depth=15,               # Deeper trees for complex patterns
                        min_samples_split=5,        # Prevent overfitting
                        min_samples_leaf=2,         # More variation in predictions
                        random_state=seed,          # Vary randomness
                        bootstrap=True,             # Enable bootstrap sampling
                        max_features='sqrt',        # Add feature randomness
                        n_jobs=-1
                    )
                    rf.fit(X_train_filled, y_train_filled)
                    rf_ensemble.append(rf)
                    
                    # Get individual predictions using filled validation data
                    rf_pred_single = rf.predict(X_val_filled)
                    rf_predictions.append(rf_pred_single)
                
                # Average ensemble predictions
                rf_pred = np.mean(rf_predictions, axis=0)
                rf_mae = mean_absolute_error(y_val_filled, rf_pred)
                rf_rmse = np.sqrt(mean_squared_error(y_val_filled, rf_pred))
                
                # Store ensemble models and performance
                rf_models[region] = rf_ensemble
                
                rf_performance[region] = {
                    'validation_mae': rf_mae,
                    'validation_rmse': rf_rmse,
                    'feature_count': len(feature_cols)
                }
                
                # Feature importance
                self.feature_importance[f"{region}_rf"] = dict(zip(feature_cols, rf.feature_importances_))
                
                print(f"TRAINED {region} RF MAE: {rf_mae:.3f}")
                
            except ValueError as e:
                print(f"ERROR Random Forest training failed for {region} - data issue: {e}")
            except Exception as e:
                print(f"ERROR Random Forest training failed for {region} - unexpected issue: {e}")
        
        self.trained_models['random_forest'] = rf_models
        self.model_performance['random_forest'] = rf_performance
        
        print(f"\nRandom Forest Training Complete: {len(rf_models)} models trained")

    # Gradient Boosting training removed - using Random Forest only

    # Linear regression methods removed - not in top 3 performers
    # (Linear, Ridge, Lasso, ElasticNet, Polynomial all perform worse than RF/GB/ARIMA)

    def evaluate_models(self):
        """Evaluate Random Forest models on test set"""
        print("\nEvaluating Random Forest Models on Test Set...")
        
        test_performance = {}
        
        for target_col in self.target_columns:
            region = self.extract_region_from_column(target_col)
            test_performance[region] = {}
            
            try:
                # Get test data
                X_test, y_test, _ = self.prepare_features(self.test_data, target_col)
                
                if y_test.isna().sum() > len(y_test) * 0.5:
                    print(f"WARNING Insufficient test data for {region}")
                    continue
                
                # Test Random Forest
                if region in self.trained_models.get('random_forest', {}):
                    rf_model = self.trained_models['random_forest'][region]
                    
                    # Handle ensemble model (list) or single model
                    if isinstance(rf_model, list):  # Ensemble model
                        # Get predictions from each model in ensemble
                        ensemble_predictions = []
                        for individual_model in rf_model:
                            pred = individual_model.predict(X_test)
                            ensemble_predictions.append(pred)
                        # Average ensemble predictions
                        rf_pred = np.mean(ensemble_predictions, axis=0)
                    else:  # Single model
                        rf_pred = rf_model.predict(X_test)
                    
                    test_performance[region]['random_forest'] = {
                        'mae': mean_absolute_error(y_test, rf_pred),
                        'rmse': np.sqrt(mean_squared_error(y_test, rf_pred))
                    }
                
                print(f"EVALUATED {region} Random Forest test evaluation complete")
                
            except KeyError as e:
                print(f"ERROR Test evaluation failed for {region} - missing data/model: {e}")
            except Exception as e:
                print(f"ERROR Test evaluation failed for {region} - unexpected issue: {e}")
        
        self.model_performance['test_results'] = test_performance
        return test_performance

    def save_random_forest_models(self):
        """Save Random Forest models with compression"""
        print("\nSaving Random Forest Models...")
        saved_models = {}
        
        # Save Random Forest models for each region
        for region in self.target_columns:
            region_name = self.extract_region_from_column(region)
            
            # Save Random Forest model if available
            if (region_name in self.trained_models.get('random_forest', {})):
                model = self.trained_models['random_forest'][region_name]
                performance = self.model_performance['random_forest'][region_name]['validation_mae']
                
                model_file = self.models_dir / f"random_forest_{region_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace("'", '')}.joblib"
                joblib.dump(model, model_file, compress=3)  # Compression level 3
                saved_models[region] = {
                    'model': 'random_forest',
                    'validation_mae': performance
                }
                print(f"SAVED: Random Forest for {region_name} (MAE: {performance:.4f})")
        
        # Update training summary with saved models
        self.saved_models_summary = saved_models
        
        # Save feature columns for each target for forecasting
        feature_columns_data = {}
        for target_col in self.target_columns:
            region_name = self.extract_region_from_column(target_col)
            if region_name in self.trained_models.get('random_forest', {}):
                # Get features from training data preparation
                X_train, y_train, feature_cols = self.prepare_features(self.train_data, target_col)
                feature_columns_data[target_col] = feature_cols
        
        # Save feature columns to file
        feature_file = self.models_dir / "feature_columns.json"
        with open(feature_file, 'w') as f:
            json.dump(feature_columns_data, f, indent=2)
        print(f"SAVED: Feature columns for {len(feature_columns_data)} targets")
        
        # Save performance metrics as JSON
        performance_file = self.models_dir / "model_evaluation_report.json"
        with open(performance_file, 'w') as f:
            json.dump(self.model_performance, f, indent=2, default=str)
        print(f"SAVED performance report: {performance_file}")
        
        # Also save as CSV for Power BI compatibility
        try:
            csv_df = self.convert_performance_to_csv(self.model_performance)
            csv_file = self.models_dir / "model_evaluation_flat.csv"
            csv_df.to_csv(csv_file, index=False)
            print(f"SAVED CSV performance report: {csv_file}")
            print(f"CSV contains {len(csv_df)} rows for Power BI import")
            
            # Create CSV file for Random Forest only
            output_dir = self.models_dir / "evaluation_csvs"
            output_dir.mkdir(exist_ok=True)
            
            if 'random_forest' in self.model_performance:
                rf_df = csv_df[csv_df['Algorithm'] == 'random_forest'].copy()
                rf_file = output_dir / 'random_forest_evaluation.csv'
                rf_df.to_csv(rf_file, index=False)
                print(f"SAVED Random Forest CSV: {rf_file} ({len(rf_df)} rows)")
                    
        except Exception as e:
            print(f"WARNING: Could not create CSV files: {e}")
            print("JSON file saved successfully, CSV generation failed")
        
        # Save feature importance for Random Forest models
        if hasattr(self, 'feature_importance') and self.feature_importance:
            importance_file = self.models_dir / "feature_importance.json"
            with open(importance_file, 'w') as f:
                json.dump(self.feature_importance, f, indent=2, default=str)
            print(f"SAVED feature importance: {importance_file}")
        
        print(f"SAVED: {len(saved_models)} Random Forest models with compression")

    def generate_summary_report(self):
        """Generate executive summary for Random Forest models"""
        print("\nGenerating Random Forest Summary Report...")
        
        summary = {
            "training_date": datetime.now().isoformat(),
            "target_regions": self.target_regions,
            "models_trained": ['random_forest'],
            "model_info": {
                "algorithm": "Random Forest Regressor",
                "all_regions": "Random Forest used for all regions"
            },
            "dataset_info": {
                "train_records": len(self.train_data),
                "validation_records": len(self.validation_data),
                "test_records": len(self.test_data),
                "total_features": self.feature_summary.get('total_features', len(self.train_data.columns) - 1)
            },
            "models_by_region": {}
        }
        
        # Use saved Random Forest models summary
        if hasattr(self, 'saved_models_summary'):
            summary["models_by_region"] = self.saved_models_summary
        else:
            # Fallback: Get Random Forest model performance for each region
            for target_col in self.target_columns:
                region = self.extract_region_from_column(target_col)
                
                if ('random_forest' in self.model_performance and 
                    region in self.model_performance['random_forest']):
                    mae = self.model_performance['random_forest'][region].get('validation_mae', None)
                    summary["models_by_region"][target_col] = {
                        "model": "random_forest",
                        "validation_mae": mae
                    }
        
        # Save summary
        summary_file = self.models_dir / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"SAVED Random Forest Training Summary: {summary_file}")
        return summary

    def convert_performance_to_csv(self, performance_data):
        """
        Convert the model performance data to CSV format for Power BI compatibility.
        Flattens the nested JSON structure into a tabular format.
        """
        rows = []
        
        # Process Random Forest algorithm
        for algorithm in performance_data:
            if algorithm in ['random_forest']:
                algorithm_data = performance_data[algorithm]
                
                # Process each demographic within the algorithm
                for demographic, metrics in algorithm_data.items():
                    row = {
                        'Algorithm': algorithm,
                        'Demographic': demographic,
                        'Series_Name': demographic.replace('_', ' ').title()
                    }
                    
                    # Random Forest metrics
                    row['Order_P'] = None
                    row['Order_D'] = None
                    row['Order_Q'] = None
                    row['AIC'] = None
                    row['Validation_MAE'] = metrics.get('validation_mae', None)
                    row['Validation_RMSE'] = metrics.get('validation_rmse', None)
                    row['Validation_MAPE'] = None  # Random Forest doesn't have MAPE
                    row['Feature_Count'] = metrics.get('feature_count', None)
                    
                    rows.append(row)
            
            # Handle test_results if present
            elif algorithm == 'test_results':
                test_data = performance_data[algorithm]
                for demographic, models in test_data.items():
                    for model_type, test_metrics in models.items():
                        row = {
                            'Algorithm': f"{model_type}_test",
                            'Demographic': demographic,
                            'Series_Name': demographic.replace('_', ' ').title(),
                            'Order_P': None,
                            'Order_D': None,
                            'Order_Q': None,
                            'AIC': None,
                            'Validation_MAE': test_metrics.get('mae', None),
                            'Validation_RMSE': test_metrics.get('rmse', None),
                            'Validation_MAPE': None,
                            'Feature_Count': None
                        }
                        rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Sort by Algorithm and Demographic for better organization
        df = df.sort_values(['Algorithm', 'Demographic'])
        
        return df

    # NOTE: Forecasting functionality moved to unemployment_forecaster_fixed.py
    # This separation ensures methodological correctness by preventing data leakage
    # 
    # IMPORTANT: After fixing data leakage, ensemble models (RF/GB/LSTM) may fail to train
    # due to reduced feature availability and NaN values. This is EXPECTED behavior.
    # ARIMA models are more robust to sparse data and remain the primary forecasting method.
    #
    # The old generate_forecasts() method had critical flaws (now fixed in separate script):
    # 1. Used test_data.tail(1) - accessing future data (data leakage)
    # 2. Generated static forecasts by repeating single predictions  
    # 3. Did not implement proper multi-step forecasting methodology

    def run_random_forest_training_pipeline(self):
        """Execute Random Forest model training pipeline"""
        print("Starting Random Forest Model Training Pipeline...")
        print("MODEL: Random Forest Regressor for all regions")
        print("=" * 70)
        
        # Step 1: Load Data
        if not self.load_datasets():
            print("ERROR Failed to load datasets. Exiting.")
            return False
        
        print(f"DATA LOADED: Training on {len(self.target_columns)} regions with {len(self.train_data)} samples")
        
        # Step 2: Random Forest Model Training
        print("\nSTEP 2: RANDOM FOREST TRAINING")
        print("Training Random Forest models for all regions...")
        
        start_time = datetime.now()
        
        # Train Random Forest models
        self.train_random_forest_models()
        
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()
        
        print(f"TRAINING COMPLETED in {training_duration:.1f} seconds")
        
        # Step 3: Evaluate Performance
        print("\nSTEP 3: MODEL EVALUATION")
        self.evaluate_models()
        
        # Step 4: Save Random Forest Models
        print("\nSTEP 4: MODEL STORAGE")
        self.save_random_forest_models()
        
        # Step 5: Generate Summary
        summary = self.generate_summary_report()
        
        print(f"\nTRAINING SUMMARY:")
        print(f"- Training Time: {training_duration:.1f}s")
        print(f"- Algorithm: Random Forest Regressor")
        print(f"- Compression: Level 3 applied to all model files")
        print(f"- Models: {len(self.target_columns)} regions trained")
        
        # Forecasting moved to separate script
        print("\nNOTE: For forecasting, use: python unemployment_forecaster_fixed.py")
        print("(Forecasting removed from trainer to prevent data leakage)")
        
        print("\n" + "=" * 60)
        print("RANDOM FOREST TRAINING PIPELINE COMPLETE!")
        print("=" * 60)
        
        # Display summary
        print("\nTRAINING SUMMARY:")
        print(f"- Regions: {', '.join(self.target_regions)}")
        print(f"- Model: {', '.join(summary['models_trained'])}")
        print(f"- Training Records: {summary['dataset_info']['train_records']}")
        print(f"- Algorithm: {summary['model_info']['algorithm']}")
        
        print("\nMODELS BY REGION:")
        for region, info in summary["models_by_region"].items():
            if info.get('model'):
                print(f"- {region}: {info['model']} (MAE: {info['validation_mae']:.3f})")
        
        print(f"\nAll results saved to: {self.models_dir}")
        print("Ready for dashboard integration and MBIE presentation!")
        
        return True


def main():
    """Main execution function - Random Forest Version"""
    print("NZ UNEMPLOYMENT FORECASTING MODEL TRAINER - RANDOM FOREST")
    print("Team JRKI - Production v2.2 Random Forest")
    print("ALGORITHM: Random Forest Regressor for all regions")
    print("=" * 70)
    
    # Initialize trainer
    trainer = UnemploymentModelTrainer()
    
    # Run Random Forest pipeline
    success = trainer.run_random_forest_training_pipeline()
    
    if success:
        print("\n[SUCCESS] Random Forest model training completed!")
        print("BENEFITS: Simplified architecture, consistent algorithm across regions")
        print("Ready for forecasting dashboard and client presentation.")
    else:
        print("\n[ERROR] Random Forest model training encountered issues.")
        print("Check data files and try again.")


if __name__ == "__main__":
    main()