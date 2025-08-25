#!/usr/bin/env python3
"""
Simple Quarterly Update Orchestrator
NZ Unemployment Forecasting System - Production Pipeline

Streamlined orchestration for quarterly data updates and model refresh.
"""

import pandas as pd
import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import shutil

class SimpleOrchestrator:
    """Simple quarterly update orchestrator with minimal complexity"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.resolve()
        self.backup_dir = self.base_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # Essential scripts only - using original cleaner for robustness
        self.scripts = [
            'comprehensive_data_cleaner.py',
            'time_series_aligner_simplified.py', 
            'temporal_data_splitter.py',
            'unemployment_model_trainer.py',
            'unemployment_forecaster_fixed.py',
            'demographic_unemployment_analyzer.py'
        ]
    
    def create_backup(self):
        """Robust backup of critical files with error handling"""
        print("Creating backup...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.backup_dir / f"backup_{timestamp}"
        backup_path.mkdir(exist_ok=True)
        
        backup_success = True
        files_backed_up = 0
        
        # Backup models directory
        models_dir = self.base_dir / "models"
        if models_dir.exists():
            try:
                # Check if directory has content
                model_files = list(models_dir.glob("*"))
                if model_files:
                    shutil.copytree(models_dir, backup_path / "models", dirs_exist_ok=True)
                    files_backed_up += len(model_files)
                    print(f"Backed up {len(model_files)} model files")
                else:
                    print("WARNING: Models directory is empty - skipping backup")
            except Exception as e:
                print(f"ERROR: Failed to backup models directory: {e}")
                backup_success = False
        else:
            print("WARNING: Models directory does not exist - skipping backup")
        
        # Backup model_ready_data directory
        data_dir = self.base_dir / "model_ready_data"
        if data_dir.exists():
            try:
                # Check if directory has content
                data_files = list(data_dir.glob("*"))
                if data_files:
                    shutil.copytree(data_dir, backup_path / "model_ready_data", dirs_exist_ok=True)
                    files_backed_up += len(data_files)
                    print(f"Backed up {len(data_files)} data files")
                else:
                    print("WARNING: Model ready data directory is empty - skipping backup")
            except Exception as e:
                print(f"ERROR: Failed to backup model ready data directory: {e}")
                backup_success = False
        else:
            print("WARNING: Model ready data directory does not exist - skipping backup")
        
        # If no files were backed up, mark backup as potentially problematic
        if files_backed_up == 0:
            print("WARNING: No files were backed up - this may indicate a problem")
            # Create a status file to indicate empty backup
            with open(backup_path / "backup_status.txt", "w") as f:
                f.write(f"Backup created: {timestamp}\n")
                f.write("Status: EMPTY - No files found to backup\n")
                f.write(f"Models dir exists: {models_dir.exists()}\n")
                f.write(f"Data dir exists: {data_dir.exists()}\n")
        else:
            # Create a status file with successful backup info
            with open(backup_path / "backup_status.txt", "w") as f:
                f.write(f"Backup created: {timestamp}\n")
                f.write(f"Status: SUCCESS - {files_backed_up} files backed up\n")
                f.write(f"Models dir exists: {models_dir.exists()}\n")
                f.write(f"Data dir exists: {data_dir.exists()}\n")
        
        if backup_success and files_backed_up > 0:
            print(f"Backup created successfully: {backup_path}")
        else:
            print(f"Backup created with warnings: {backup_path}")
        
        return backup_path
    
    def cleanup_old_backups(self, keep_count=10):
        """Clean up old backups, keeping only the most recent ones"""
        if not self.backup_dir.exists():
            return
        
        # Get all backup directories sorted by creation time
        backup_dirs = []
        for backup_dir in self.backup_dir.glob("backup_*"):
            if backup_dir.is_dir():
                backup_dirs.append((backup_dir.stat().st_mtime, backup_dir))
        
        # Sort by modification time (newest first)
        backup_dirs.sort(reverse=True)
        
        # Keep only the most recent backups
        if len(backup_dirs) > keep_count:
            backups_to_remove = backup_dirs[keep_count:]
            print(f"Cleaning up {len(backups_to_remove)} old backups...")
            
            for _, backup_path in backups_to_remove:
                try:
                    # Check if it's an empty backup before deletion
                    is_empty = not any(backup_path.glob("models/*")) and not any(backup_path.glob("model_ready_data/*"))
                    if is_empty:
                        print(f"Removing empty backup: {backup_path.name}")
                    else:
                        print(f"Removing old backup: {backup_path.name}")
                    
                    shutil.rmtree(backup_path)
                except Exception as e:
                    print(f"Failed to remove backup {backup_path}: {e}")
        else:
            print(f"No cleanup needed - {len(backup_dirs)} backups (keeping {keep_count})")
    
    def run_script(self, script_name):
        """Run a pipeline script"""
        script_path = self.base_dir / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        
        print(f"Running {script_name}...")
        result = subprocess.run([sys.executable, str(script_path)], 
                              cwd=self.base_dir, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error in {script_name}: {result.stderr}")
            return False
        return True
    
    def verify_output(self):
        """Simple verification of key outputs"""
        print("Verifying outputs...")
        
        # Check cleaned data exists
        data_cleaned_dir = self.base_dir / 'data_cleaned'
        if not data_cleaned_dir.exists() or not any(data_cleaned_dir.glob("cleaned_*.csv")):
            print("ERROR: Cleaned data files not found")
            return False
        
        # Check integrated dataset (created by aligner)
        integrated_path = data_cleaned_dir / 'integrated_forecasting_dataset.csv'
        if not integrated_path.exists():
            print("ERROR: Integrated dataset not found")
            return False
        
        # Check models
        models_dir = self.base_dir / 'models'
        if not models_dir.exists() or not (models_dir / 'fixed_unemployment_forecasts.json').exists():
            print("ERROR: Forecasts not found")
            return False
            
        print("All outputs verified successfully")
        return True
    
    def run_quarterly_update(self):
        """Execute complete quarterly update"""
        print("=" * 60)
        print("SIMPLE QUARTERLY UPDATE ORCHESTRATOR")
        print("=" * 60)
        
        try:
            # Step 1: Backup
            backup_path = self.create_backup()
            
            # Step 2: Run pipeline scripts
            for script in self.scripts:
                if not self.run_script(script):
                    raise Exception(f"Script failed: {script}")
            
            # Step 3: Verify
            if not self.verify_output():
                raise Exception("Output verification failed")
            
            # Step 4: Cleanup old backups
            self.cleanup_old_backups(keep_count=10)
            
            print("\n[SUCCESS] Quarterly update completed successfully")
            return True
            
        except Exception as e:
            print(f"\n[ERROR] Quarterly update failed: {e}")
            print(f"Backup available for rollback: {backup_path}")
            return False

def main():
    orchestrator = SimpleOrchestrator()
    success = orchestrator.run_quarterly_update()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()