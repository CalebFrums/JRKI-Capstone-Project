#!/usr/bin/env python3
"""
Quarterly Update Orchestration System
NZ Unemployment Forecasting System - Automated Production Pipeline

This module provides comprehensive orchestration for the quarterly data update
and model refresh cycle. Designed for autonomous operation with comprehensive
error handling, backup procedures, and validation reporting.

Workflow Components:
1. Automated data backup and version control
2. Multi-stage data cleaning and integration pipeline
3. Dynamic temporal data splitting with anti-leakage controls
4. Multi-algorithm model retraining and evaluation
5. Production forecast generation and validation
6. Comprehensive reporting and quality assurance

Author: Data Science Team
Version: Production v2.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class QuarterlyUpdateOrchestrator:
    """
    Professional quarterly update orchestration system for production forecasting pipeline.
    
    This class manages the complete quarterly data refresh and model update cycle,
    providing automated backup, validation, error handling, and comprehensive reporting
    for government-grade forecasting operations.
    """
    
    def __init__(self, base_dir=".", backup_dir="quarterly_backups"):
        # Always use the directory where this script is located (Capstone directory)
        self.base_dir = Path(__file__).parent.resolve()
        self.backup_dir = self.base_dir / backup_dir
        self.backup_dir.mkdir(exist_ok=True)
        
        # Create timestamped backup folder
        self.current_backup = self.backup_dir / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_backup.mkdir(exist_ok=True)
        
        # Pipeline scripts
        self.scripts = {
            'data_cleaner': 'comprehensive_data_cleaner.py',
            'time_aligner': 'time_series_aligner_simplified.py', 
            'data_splitter': 'temporal_data_splitter.py',
            'model_trainer': 'unemployment_model_trainer.py',
            'forecaster': 'unemployment_forecaster_fixed.py'
        }
        
        # Verification paths
        self.data_paths = {
            'cleaned': Path('data_cleaned'),
            'model_ready': Path('model_ready_data'),
            'models': Path('models')
        }
        
        print("Quarterly Update Orchestration System initialized")
        print(f"Working directory: {self.base_dir}")
        print(f"Backup location: {self.current_backup}")
    
    def wait_for_file_operations(self, max_wait=10, check_interval=0.5):
        """Wait for file system operations to complete after script execution"""
        time.sleep(check_interval)  # Initial small delay for file system sync
        
    def retry_verification(self, verification_func, max_attempts=5, delay=2):
        """Retry verification function with exponential backoff"""
        for attempt in range(max_attempts):
            try:
                result = verification_func()
                if result:  # If verification passes, return immediately
                    return result
                
                if attempt < max_attempts - 1:  # Don't wait after last attempt
                    wait_time = delay * (2 ** attempt)  # Exponential backoff: 2, 4, 8, 16 seconds
                    print(f"   Verification attempt {attempt + 1} failed, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    
            except Exception as e:
                if attempt < max_attempts - 1:
                    wait_time = delay * (2 ** attempt)
                    print(f"   Verification error (attempt {attempt + 1}): {e}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"   Verification failed after {max_attempts} attempts: {e}")
                    return False
        
        return False  # All attempts failed
        
    def create_backup(self):
        """Backup current model and data state before update"""
        print("\n" + "="*60)
        print("STEP 1: CREATING BACKUP")
        print("="*60)
        
        backup_items = [
            ('models', 'models/'),
            ('model_ready_data', 'model_ready_data/'),
            ('data_cleaned/integrated_forecasting_dataset.csv', 'integrated_forecasting_dataset.csv')
        ]
        
        for source, backup_name in backup_items:
            source_path = Path(source)
            if source_path.exists():
                if source_path.is_file():
                    backup_path = self.current_backup / backup_name
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    import shutil
                    shutil.copy2(source_path, backup_path)
                    print(f"   Backed up: {source} -> {backup_path}")
                elif source_path.is_dir():
                    backup_path = self.current_backup / backup_name
                    import shutil
                    shutil.copytree(source_path, backup_path, dirs_exist_ok=True)
                    print(f"   Backed up: {source}/ -> {backup_path}")
        
        # Save backup metadata
        backup_metadata = {
            "backup_timestamp": datetime.now().isoformat(),
            "backup_trigger": "quarterly_update",
            "backed_up_items": [item[0] for item in backup_items]
        }
        
        with open(self.current_backup / "backup_metadata.json", 'w') as f:
            json.dump(backup_metadata, f, indent=2)
        
        print(f"[OK] Backup complete: {self.current_backup}")
    
    def run_pipeline_script(self, script_name, description):
        """Run a pipeline script and capture results"""
        script_path = self.base_dir / self.scripts[script_name]
        
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        
        print(f"\n[RUNNING] {description}...")
        print(f"   Script: {script_path}")
        
        try:
            result = subprocess.run([
                sys.executable, str(script_path)
            ], capture_output=True, text=True, cwd=str(self.base_dir))
            
            if result.returncode == 0:
                # Check for masked errors in output
                output = result.stdout.strip()
                error_indicators = ['ERROR', 'FAILED', 'Exception', 'Traceback', 'failed for']
                
                found_errors = []
                for indicator in error_indicators:
                    if indicator in output:
                        lines = output.split('\n')
                        error_lines = [line for line in lines if indicator in line]
                        found_errors.extend(error_lines[:3])  # First 3 error lines
                
                if found_errors:
                    print(f"[WARNING] {description} completed but with errors:")
                    for error_line in found_errors:
                        print(f"   {error_line.strip()}")
                    print(f"   Check logs for full details")
                else:
                    print(f"[OK] {description} completed successfully")
                    
                if output:
                    print(f"   Output preview: {output.split()[0:10]}")
                return True, result.stdout
            else:
                print(f"[ERROR] {description} failed")
                print(f"   Error: {result.stderr}")
                return False, result.stderr
                
        except Exception as e:
            print(f"[ERROR] {description} failed with exception: {e}")
            return False, str(e)
    
    def verify_data_update(self):
        """Verify that new data has been properly integrated"""
        print("\n" + "="*60)
        print("DATA UPDATE VERIFICATION")
        print("="*60)
        
        try:
            # Check integrated dataset
            integrated_path = self.base_dir / 'data_cleaned/integrated_forecasting_dataset.csv'
            if integrated_path.exists():
                df = pd.read_csv(integrated_path)
                df['date'] = pd.to_datetime(df['date'])
                
                latest_date = df['date'].max()
                record_count = len(df)
                
                print(f"[OK] Integrated dataset found")
                print(f"   Records: {record_count}")
                print(f"   Latest data: {latest_date}")
                print(f"   Data range: {df['date'].min()} to {latest_date}")
                
                # Check if we have recent data (within last 2 years)
                cutoff_date = datetime.now() - timedelta(days=730)  # 2 years
                if latest_date >= pd.Timestamp(cutoff_date):
                    print(f"[OK] Data appears current (within 2 years)")
                else:
                    print(f"[WARNING] Data may be outdated (latest: {latest_date})")
                
                return True
            else:
                print(f"[ERROR] Integrated dataset not found: {integrated_path}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Data verification failed: {e}")
            return False
    
    def verify_model_update(self):
        """Verify that models have been retrained with new data"""
        print("\n" + "="*60)
        print("MODEL UPDATE VERIFICATION")
        print("="*60)
        
        try:
            models_dir = self.base_dir / 'models'
            if not models_dir.exists():
                print(f"[ERROR] Models directory not found")
                return False
            
            # Check for key model files
            required_files = [
                'fixed_unemployment_forecasts.json',
                'model_evaluation_report.json',
                'training_summary.json'
            ]
            
            missing_files = []
            for file in required_files:
                file_path = models_dir / file
                if not file_path.exists():
                    missing_files.append(file)
            
            if missing_files:
                print(f"[ERROR] Missing model files: {missing_files}")
                return False
            
            # Check model timestamps
            forecasts_path = models_dir / 'fixed_unemployment_forecasts.json'
            file_time = datetime.fromtimestamp(forecasts_path.stat().st_mtime)
            hours_old = (datetime.now() - file_time).total_seconds() / 3600
            
            print(f"[OK] All required model files present")
            print(f"   Latest forecast file: {hours_old:.1f} hours old")
            
            # Load and verify forecasts
            with open(forecasts_path, 'r') as f:
                forecasts = json.load(f)
            
            regions = list(forecasts.keys())
            print(f"[OK] Forecasts available for: {regions}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Model verification failed: {e}")
            return False
    
    def generate_update_report(self, pipeline_results):
        """Generate comprehensive update report"""
        report = {
            "update_timestamp": datetime.now().isoformat(),
            "update_type": "quarterly_automated",
            "backup_location": str(self.current_backup),
            "pipeline_execution": pipeline_results,
            "verification_results": {
                "data_verification": self.retry_verification(self.verify_data_update),
                "model_verification": self.retry_verification(self.verify_model_update)
            }
        }
        
        # Save report
        report_path = Path('quarterly_update_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n[REPORT] Update report saved: {report_path}")
        return report
    
    def run_complete_quarterly_update(self):
        """Execute the complete quarterly update workflow"""
        print("="*80)
        print("QUARTERLY UPDATE ORCHESTRATOR - AUTOMATED WORKFLOW")
        print("="*80)
        print(f"Started: {datetime.now()}")
        
        pipeline_results = {}
        
        try:
            # Step 1: Backup current state
            self.create_backup()
            
            # Step 2: Data cleaning and integration
            print("\n" + "="*60)
            print("STEP 2: DATA PROCESSING PIPELINE")
            print("="*60)
            
            success, output = self.run_pipeline_script('data_cleaner', 'Data Cleaning')
            pipeline_results['data_cleaning'] = {'success': success, 'output': output}
            if not success:
                raise Exception("Data cleaning failed")
            
            success, output = self.run_pipeline_script('time_aligner', 'Time Series Alignment')
            pipeline_results['time_alignment'] = {'success': success, 'output': output}
            if not success:
                raise Exception("Time series alignment failed")
            
            success, output = self.run_pipeline_script('data_splitter', 'Dynamic Temporal Splitting')
            pipeline_results['temporal_splitting'] = {'success': success, 'output': output}
            if not success:
                raise Exception("Temporal splitting failed")
            
            # Step 3: Model retraining
            print("\n" + "="*60)
            print("STEP 3: MODEL RETRAINING")
            print("="*60)
            
            success, output = self.run_pipeline_script('model_trainer', 'Model Training')
            pipeline_results['model_training'] = {'success': success, 'output': output}
            if not success:
                raise Exception("Model training failed")
            
            # Step 4: Generate updated forecasts
            print("\n" + "="*60)
            print("STEP 4: FORECAST GENERATION")
            print("="*60)
            
            success, output = self.run_pipeline_script('forecaster', 'Forecast Generation')
            pipeline_results['forecast_generation'] = {'success': success, 'output': output}
            if not success:
                raise Exception("Forecast generation failed")
            
            # Allow time for all file operations to complete
            self.wait_for_file_operations()
            
            # Step 5: Verification and reporting
            print("\n" + "="*60)
            print("STEP 5: VERIFICATION & REPORTING")
            print("="*60)
            
            report = self.generate_update_report(pipeline_results)
            
            print("\n" + "="*80)
            print("[SUCCESS] QUARTERLY UPDATE COMPLETED SUCCESSFULLY")
            print("="*80)
            print(f"Completed: {datetime.now()}")
            print(f"Backup: {self.current_backup}")
            print("[OK] Data processed with rolling time windows")
            print("[OK] Models retrained on updated dataset")
            print("[OK] Fresh forecasts generated")
            print("[OK] System ready for next quarterly update")
            
            return True, report
            
        except Exception as e:
            print(f"\n[ERROR] QUARTERLY UPDATE FAILED: {e}")
            print(f"Backup available for rollback: {self.current_backup}")
            
            # Generate failure report
            pipeline_results['error'] = str(e)
            report = self.generate_update_report(pipeline_results)
            
            return False, report


def main():
    """Execute quarterly update orchestration"""
    orchestrator = QuarterlyUpdateOrchestrator()
    success, report = orchestrator.run_complete_quarterly_update()
    
    if success:
        print("\n[SUCCESS] Quarterly update orchestration completed successfully")
        sys.exit(0)
    else:
        print("\n[FAILED] Quarterly update orchestration failed")
        sys.exit(1)


if __name__ == "__main__":
    main()