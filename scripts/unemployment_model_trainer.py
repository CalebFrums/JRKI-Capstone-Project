#!/usr/bin/env python3
"""
Unemployment Forecasting Model Trainer
NZ Unemployment Forecasting Project - Week 7 Final Implementation

Creates ARIMA, LSTM, and ensemble models for unemployment forecasting
using the feature-engineered datasets for Auckland, Wellington, Canterbury.

Author: Team JRKI - Final Week 7 Model Training
Date: Week 7 Implementation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
from datetime import datetime
import pickle

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# Neural Network Libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("WARNING TensorFlow not available - LSTM models will be skipped")
    TENSORFLOW_AVAILABLE = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class UnemploymentModelTrainer:
    """
    Trains multiple forecasting models for NZ unemployment data
    Focuses on Auckland, Wellington, Canterbury regions
    """
    
    def __init__(self, data_dir="model_ready_data", models_dir="models", config_file="simple_config.json"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Load target regions from config
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            self.target_regions = config.get('regions', {}).get('unemployment_core', ['Auckland', 'Wellington', 'Canterbury'])
        except:
            self.target_regions = ['Auckland', 'Wellington', 'Canterbury']
        
        self.target_columns = [f"{region}_Male_unemployment_rate" for region in self.target_regions]
        self.trained_models = {}
        self.model_performance = {}
        self.feature_importance = {}
        
        print("Unemployment Model Trainer Initialized")
        print(f"Target Regions: {', '.join(self.target_regions)}")
        print(f"Data Directory: {self.data_dir}")
        print(f"Models Directory: {self.models_dir}")

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
            
            return True
            
        except Exception as e:
            print(f"ERROR loading datasets: {e}")
            return False

    def prepare_features(self, dataset, target_col):
        """Prepare features for modeling by removing target and unnecessary columns"""
        # Remove target columns and non-predictive features
        exclude_cols = self.target_columns + ['date', 'quarter', 'year']
        feature_cols = [col for col in dataset.columns if col not in exclude_cols]
        
        X = dataset[feature_cols].ffill().fillna(0)
        y = dataset[target_col].ffill()
        
        return X, y, feature_cols

    def train_arima_models(self):
        """Train ARIMA models for each target region"""
        print("\nTraining ARIMA Models...")
        
        arima_models = {}
        arima_performance = {}
        
        for target_col in self.target_columns:
            region = target_col.replace('_Male_unemployment_rate', '')
            print(f"\nTraining ARIMA for {region}...")
            
            try:
                # Prepare time series data
                train_data_copy = self.train_data.copy()
                validation_data_copy = self.validation_data.copy()
                
                # Ensure date column is datetime
                train_data_copy['date'] = pd.to_datetime(train_data_copy['date'])
                validation_data_copy['date'] = pd.to_datetime(validation_data_copy['date'])
                
                train_series = train_data_copy.set_index('date')[target_col].dropna().sort_index()
                validation_series = validation_data_copy.set_index('date')[target_col].dropna().sort_index()
                
                if len(train_series) < 20:  # Need minimum data for ARIMA
                    print(f"WARNING Insufficient data for {region} ARIMA model")
                    continue
                
                # Auto ARIMA parameter selection (simple approach)
                best_aic = np.inf
                best_order = (1, 1, 1)
                
                for p in range(0, 3):
                    for d in range(0, 2):
                        for q in range(0, 3):
                            try:
                                model = ARIMA(train_series, order=(p, d, q))
                                fitted_model = model.fit()
                                if fitted_model.aic < best_aic:
                                    best_aic = fitted_model.aic
                                    best_order = (p, d, q)
                            except:
                                continue
                
                # Train final model with best parameters
                final_model = ARIMA(train_series, order=best_order)
                fitted_arima = final_model.fit()
                
                # Validate on validation set
                forecast = fitted_arima.forecast(steps=len(validation_series))
                mae = mean_absolute_error(validation_series, forecast)
                rmse = np.sqrt(mean_squared_error(validation_series, forecast))
                mape = mean_absolute_percentage_error(validation_series, forecast)
                
                arima_models[region] = fitted_arima
                arima_performance[region] = {
                    'order': best_order,
                    'aic': best_aic,
                    'validation_mae': mae,
                    'validation_rmse': rmse,
                    'validation_mape': mape
                }
                
                print(f"TRAINED {region} ARIMA({best_order[0]},{best_order[1]},{best_order[2]}) - MAE: {mae:.3f}")
                
            except Exception as e:
                print(f"ERROR ARIMA training failed for {region}: {e}")
        
        self.trained_models['arima'] = arima_models
        self.model_performance['arima'] = arima_performance
        print(f"\nARIMA Training Complete: {len(arima_models)} models trained")

    def prepare_lstm_sequences(self, data, target_col, sequence_length=12, scaler_X=None, scaler_y=None):
        """Prepare sequences for LSTM training"""
        # Get features and target
        X, y, feature_cols = self.prepare_features(data, target_col)
        
        # Handle missing data aggressively for LSTM
        X_filled = X.ffill().bfill().fillna(X.mean())
        y_filled = y.ffill().bfill().fillna(y.mean())
        
        # Check if we still have too much missing data
        if y_filled.isna().sum() > len(y_filled) * 0.95:
            return np.array([]), np.array([]), None, None, feature_cols
        
        # Scale features - create new scalers only if not provided (training data)
        if scaler_X is None:
            scaler_X = MinMaxScaler()
            X_scaled = scaler_X.fit_transform(X_filled)
        else:
            X_scaled = scaler_X.transform(X_filled)
            
        if scaler_y is None:
            scaler_y = MinMaxScaler()
            y_scaled = scaler_y.fit_transform(y_filled.values.reshape(-1, 1)).flatten()
        else:
            y_scaled = scaler_y.transform(y_filled.values.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X_scaled)):
            X_sequences.append(X_scaled[i-sequence_length:i])
            y_sequences.append(y_scaled[i])
        
        return np.array(X_sequences), np.array(y_sequences), scaler_X, scaler_y, feature_cols

    def train_lstm_models(self):
        """Train LSTM models for each target region"""
        if not TENSORFLOW_AVAILABLE:
            print("\nWARNING TensorFlow not available - Skipping LSTM training")
            return
        
        print("\nTraining LSTM Models...")
        
        lstm_models = {}
        lstm_performance = {}
        lstm_scalers = {}
        
        for target_col in self.target_columns:
            region = target_col.replace('_Male_unemployment_rate', '')
            print(f"\nTraining LSTM for {region}...")
            
            try:
                # Prepare sequences
                X_train_seq, y_train_seq, scaler_X, scaler_y, feature_cols = self.prepare_lstm_sequences(
                    self.train_data, target_col, sequence_length=12
                )
                X_val_seq, y_val_seq, _, _, _ = self.prepare_lstm_sequences(
                    self.validation_data, target_col, sequence_length=12, 
                    scaler_X=scaler_X, scaler_y=scaler_y
                )
                
                if len(X_train_seq) < 20:  # Need minimum sequences for LSTM
                    print(f"WARNING Insufficient sequence data for {region} LSTM model")
                    continue
                
                # Build LSTM model
                model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
                    Dropout(0.2),
                    LSTM(50, return_sequences=False),
                    Dropout(0.2),
                    Dense(25),
                    Dense(1)
                ])
                
                model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
                
                # Train model
                history = model.fit(
                    X_train_seq, y_train_seq,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_val_seq, y_val_seq),
                    verbose=0
                )
                
                # Evaluate
                train_pred = model.predict(X_train_seq, verbose=0)
                val_pred = model.predict(X_val_seq, verbose=0)
                
                # Calculate metrics (unscaled)
                train_pred_unscaled = scaler_y.inverse_transform(train_pred)
                val_pred_unscaled = scaler_y.inverse_transform(val_pred.reshape(-1, 1))
                
                y_train_unscaled = scaler_y.inverse_transform(y_train_seq.reshape(-1, 1))
                y_val_unscaled = scaler_y.inverse_transform(y_val_seq.reshape(-1, 1))
                
                val_mae = mean_absolute_error(y_val_unscaled, val_pred_unscaled)
                val_rmse = np.sqrt(mean_squared_error(y_val_unscaled, val_pred_unscaled))
                
                lstm_models[region] = model
                lstm_scalers[region] = {'scaler_X': scaler_X, 'y_scaler': scaler_y}
                lstm_performance[region] = {
                    'validation_mae': val_mae,
                    'validation_rmse': val_rmse,
                    'sequence_length': 12,
                    'feature_count': len(feature_cols)
                }
                
                print(f"TRAINED {region} LSTM - MAE: {val_mae:.3f}")
                
            except Exception as e:
                print(f"ERROR LSTM training failed for {region}: {e}")
        
        self.trained_models['lstm'] = lstm_models
        self.trained_models['lstm_scalers'] = lstm_scalers
        self.model_performance['lstm'] = lstm_performance
        print(f"\nLSTM Training Complete: {len(lstm_models)} models trained")

    def train_ensemble_models(self):
        """Train Random Forest and Gradient Boosting models"""
        print("\nTraining Ensemble Models...")
        
        rf_models = {}
        gb_models = {}
        rf_performance = {}
        gb_performance = {}
        
        for target_col in self.target_columns:
            region = target_col.replace('_Male_unemployment_rate', '')
            print(f"\nTraining ensemble models for {region}...")
            
            try:
                # Prepare data
                X_train, y_train, feature_cols = self.prepare_features(self.train_data, target_col)
                X_val, y_val, _ = self.prepare_features(self.validation_data, target_col)
                
                if y_train.isna().sum() > len(y_train) * 0.95:  # Skip if >95% missing (was 50%)
                    print(f"WARNING Too much missing data for {region} ensemble models")
                    continue
                
                # Handle missing data more aggressively for ensemble models
                # Forward fill, then backward fill, then use mean for remaining NaN
                y_train_filled = y_train.ffill().bfill().fillna(y_train.mean())
                X_train_filled = X_train.ffill().bfill().fillna(X_train.mean())
                y_val_filled = y_val.ffill().bfill().fillna(y_train.mean())
                
                # Random Forest
                rf = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                rf.fit(X_train_filled, y_train_filled)
                
                rf_pred = rf.predict(X_val.ffill().bfill().fillna(X_train.mean()))
                rf_mae = mean_absolute_error(y_val, rf_pred)
                rf_rmse = np.sqrt(mean_squared_error(y_val, rf_pred))
                
                # Gradient Boosting
                gb = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
                gb.fit(X_train_filled, y_train_filled)
                
                gb_pred = gb.predict(X_val.ffill().bfill().fillna(X_train.mean()))
                gb_mae = mean_absolute_error(y_val, gb_pred)
                gb_rmse = np.sqrt(mean_squared_error(y_val, gb_pred))
                
                # Store models and performance
                rf_models[region] = rf
                gb_models[region] = gb
                
                rf_performance[region] = {
                    'validation_mae': rf_mae,
                    'validation_rmse': rf_rmse,
                    'feature_count': len(feature_cols)
                }
                
                gb_performance[region] = {
                    'validation_mae': gb_mae,
                    'validation_rmse': gb_rmse,
                    'feature_count': len(feature_cols)
                }
                
                # Feature importance
                self.feature_importance[f"{region}_rf"] = dict(zip(feature_cols, rf.feature_importances_))
                self.feature_importance[f"{region}_gb"] = dict(zip(feature_cols, gb.feature_importances_))
                
                print(f"TRAINED {region} RF MAE: {rf_mae:.3f}, GB MAE: {gb_mae:.3f}")
                
            except Exception as e:
                print(f"ERROR Ensemble training failed for {region}: {e}")
        
        self.trained_models['random_forest'] = rf_models
        self.trained_models['gradient_boosting'] = gb_models
        self.model_performance['random_forest'] = rf_performance
        self.model_performance['gradient_boosting'] = gb_performance
        
        print(f"\nEnsemble Training Complete: {len(rf_models)} RF, {len(gb_models)} GB models")

    def evaluate_models(self):
        """Evaluate all models on test set"""
        print("\nEvaluating Models on Test Set...")
        
        test_performance = {}
        
        for target_col in self.target_columns:
            region = target_col.replace('_Male_unemployment_rate', '')
            test_performance[region] = {}
            
            try:
                # Get test data
                X_test, y_test, _ = self.prepare_features(self.test_data, target_col)
                
                if y_test.isna().sum() > len(y_test) * 0.5:
                    print(f"WARNING Insufficient test data for {region}")
                    continue
                
                # Test ARIMA
                if region in self.trained_models.get('arima', {}):
                    arima_model = self.trained_models['arima'][region]
                    test_forecast = arima_model.forecast(steps=len(y_test))
                    test_performance[region]['arima'] = {
                        'mae': mean_absolute_error(y_test, test_forecast),
                        'rmse': np.sqrt(mean_squared_error(y_test, test_forecast))
                    }
                
                # Test Random Forest
                if region in self.trained_models.get('random_forest', {}):
                    rf_model = self.trained_models['random_forest'][region]
                    rf_pred = rf_model.predict(X_test)
                    test_performance[region]['random_forest'] = {
                        'mae': mean_absolute_error(y_test, rf_pred),
                        'rmse': np.sqrt(mean_squared_error(y_test, rf_pred))
                    }
                
                # Test Gradient Boosting
                if region in self.trained_models.get('gradient_boosting', {}):
                    gb_model = self.trained_models['gradient_boosting'][region]
                    gb_pred = gb_model.predict(X_test)
                    test_performance[region]['gradient_boosting'] = {
                        'mae': mean_absolute_error(y_test, gb_pred),
                        'rmse': np.sqrt(mean_squared_error(y_test, gb_pred))
                    }
                
                # Test LSTM
                if region in self.trained_models.get('lstm', {}) and TENSORFLOW_AVAILABLE:
                    lstm_model = self.trained_models['lstm'][region]
                    lstm_scalers = self.trained_models['lstm_scalers'][region]
                    
                    # Prepare test sequences
                    X_test_seq, y_test_seq, _, _, _ = self.prepare_lstm_sequences(
                        self.test_data, target_col, sequence_length=12,
                        scaler_X=lstm_scalers['scaler_X'], scaler_y=lstm_scalers['y_scaler']
                    )
                    
                    if len(X_test_seq) > 0:
                        lstm_pred = lstm_model.predict(X_test_seq, verbose=0)
                        lstm_pred_unscaled = lstm_scalers['y_scaler'].inverse_transform(lstm_pred)
                        y_test_unscaled = lstm_scalers['y_scaler'].inverse_transform(y_test_seq.reshape(-1, 1))
                        
                        test_performance[region]['lstm'] = {
                            'mae': mean_absolute_error(y_test_unscaled, lstm_pred_unscaled),
                            'rmse': np.sqrt(mean_squared_error(y_test_unscaled, lstm_pred_unscaled))
                        }
                
                print(f"EVALUATED {region} test evaluation complete")
                
            except Exception as e:
                print(f"ERROR Test evaluation failed for {region}: {e}")
        
        self.model_performance['test_results'] = test_performance
        return test_performance

    def save_models(self):
        """Save trained models and performance metrics"""
        print("\nSaving Models and Results...")
        
        # Save models
        for model_type, models in self.trained_models.items():
            for region, model in models.items():
                model_file = self.models_dir / f"{model_type}_{region.lower()}.pkl"
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)
                print(f"SAVED {model_type} model for {region}")
        
        # Save performance metrics
        performance_file = self.models_dir / "model_evaluation_report.json"
        with open(performance_file, 'w') as f:
            json.dump(self.model_performance, f, indent=2)
        print(f"SAVED performance report: {performance_file}")
        
        # Save feature importance
        if self.feature_importance:
            importance_file = self.models_dir / "feature_importance.json"
            with open(importance_file, 'w') as f:
                json.dump(self.feature_importance, f, indent=2)
            print(f"SAVED feature importance: {importance_file}")

    def generate_summary_report(self):
        """Generate executive summary of model training results"""
        print("\nGenerating Summary Report...")
        
        summary = {
            "training_date": datetime.now().isoformat(),
            "target_regions": self.target_regions,
            "models_trained": list(self.trained_models.keys()),
            "dataset_info": {
                "train_records": len(self.train_data),
                "validation_records": len(self.validation_data),
                "test_records": len(self.test_data),
                "total_features": self.feature_summary.get('total_features', 'Unknown')
            },
            "best_models_by_region": {}
        }
        
        # Find best model for each region based on validation MAE
        for target_col in self.target_columns:
            region = target_col.replace('_Male_unemployment_rate', '')
            best_mae = np.inf
            best_model = None
            
            for model_type in ['arima', 'random_forest', 'gradient_boosting']:
                if (region in self.model_performance.get(model_type, {}) and 
                    'validation_mae' in self.model_performance[model_type][region]):
                    mae = self.model_performance[model_type][region]['validation_mae']
                    if mae < best_mae:
                        best_mae = mae
                        best_model = model_type
            
            summary["best_models_by_region"][region] = {
                "best_model": best_model,
                "validation_mae": best_mae
            }
        
        # Save summary
        summary_file = self.models_dir / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"SAVED Training Summary: {summary_file}")
        return summary

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

    def run_full_training_pipeline(self):
        """Execute complete model training pipeline"""
        print("Starting Complete Model Training Pipeline...")
        print("=" * 60)
        
        # Step 1: Load Data
        if not self.load_datasets():
            print("ERROR Failed to load datasets. Exiting.")
            return False
        
        # Step 2: Train ARIMA Models
        self.train_arima_models()
        
        # Step 3: Train LSTM Models
        self.train_lstm_models()
        
        # Step 4: Train Ensemble Models
        self.train_ensemble_models()
        
        # Step 5: Evaluate on Test Set
        self.evaluate_models()
        
        # Step 6: Save Everything
        self.save_models()
        
        # Step 7: Generate Summary
        summary = self.generate_summary_report()
        
        # Step 8: Forecasting moved to separate script
        print("\nNOTE: For forecasting, use: python unemployment_forecaster_fixed.py")
        print("(Forecasting removed from trainer to prevent data leakage)")
        
        print("\n" + "=" * 60)
        print("MODEL TRAINING PIPELINE COMPLETE!")
        print("=" * 60)
        
        # Display summary
        print("\nTRAINING SUMMARY:")
        print(f"- Regions: {', '.join(self.target_regions)}")
        print(f"- Models Trained: {', '.join(summary['models_trained'])}")
        print(f"- Training Records: {summary['dataset_info']['train_records']}")
        
        print("\nBEST MODELS BY REGION:")
        for region, info in summary["best_models_by_region"].items():
            if info['best_model']:
                print(f"- {region}: {info['best_model']} (MAE: {info['validation_mae']:.3f})")
        
        print(f"\nAll results saved to: {self.models_dir}")
        print("Ready for dashboard integration and MBIE presentation!")
        
        return True


def main():
    """Main execution function"""
    print("NZ UNEMPLOYMENT FORECASTING MODEL TRAINER")
    print("Team JRKI - Final Week 7 Implementation")
    print("=" * 60)
    
    # Initialize trainer
    trainer = UnemploymentModelTrainer()
    
    # Run complete pipeline
    success = trainer.run_full_training_pipeline()
    
    if success:
        print("\nModel training completed successfully!")
        print("Ready for forecasting dashboard and client presentation.")
    else:
        print("\nERROR Model training encountered issues.")
        print("Check data files and try again.")


if __name__ == "__main__":
    main()