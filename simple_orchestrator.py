#!/usr/bin/env python3
"""
Simple Quarterly Update Orchestrator
NZ Unemployment Forecasting System - Production Pipeline

Streamlined orchestration for quarterly data updates and model refresh.
"""

import pandas as pd # type: ignore
import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import shutil
import logging
import time
import hashlib
import psutil
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import traceback

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class TaskResult:
    """Enhanced task result with 2024 MLOps tracking"""
    task_name: str
    status: TaskStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    data_lineage: Dict[str, Any] = None
    metrics: Dict[str, float] = None
    error_details: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['status'] = self.status.value
        result['start_time'] = self.start_time.isoformat()
        result['end_time'] = self.end_time.isoformat() if self.end_time else None
        return result

@dataclass
class PipelineMetrics:
    """2024 MLOps pipeline observability metrics"""
    pipeline_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration_seconds: float = 0.0
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_skipped: int = 0
    data_quality_score: float = 0.0
    model_accuracy_score: float = 0.0
    system_resource_usage: Dict[str, float] = None
    data_drift_detected: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for monitoring systems"""
        result = asdict(self)
        result['start_time'] = self.start_time.isoformat()
        result['end_time'] = self.end_time.isoformat() if self.end_time else None
        return result

class ProductionOrchestrator:
    """2024 Production-Grade MLOps Pipeline Orchestrator"""
    
    def __init__(self, config_file: str = "simple_config.json"):
        self.base_dir = Path(__file__).parent.resolve()
        self.backup_dir = self.base_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # Enhanced logging setup
        self.setup_logging()
        
        # Load configuration
        self.config = self.load_config(config_file)
        
        # Pipeline tracking
        self.pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
        self.task_results: List[TaskResult] = []
        self.pipeline_metrics = PipelineMetrics(
            pipeline_id=self.pipeline_id,
            start_time=datetime.now(),
            system_resource_usage={}
        )
        
        # Enhanced script configuration with dependencies
        self.pipeline_tasks = [
            {
                'name': 'data_cleaning',
                'script': 'comprehensive_data_cleaner.py',
                'description': 'Clean raw government datasets with 2024 quality controls',
                'timeout': 600,  # 10 minutes
                'retry_count': 2,
                'critical': True
            },
            {
                'name': 'time_series_alignment',
                'script': 'time_series_aligner_simplified.py',
                'description': 'Align and integrate time series with advanced imputation',
                'timeout': 600,  # 10 minutes
                'retry_count': 2,
                'critical': True,
                'depends_on': ['data_cleaning']
            },
            {
                'name': 'temporal_splitting',
                'script': 'temporal_data_splitter.py',
                'description': 'Split data temporally with quality gates',
                'timeout': 600,  # 10 minutes
                'retry_count': 2,
                'critical': True,
                'depends_on': ['time_series_alignment']
            },
            {
                'name': 'model_training',
                'script': 'unemployment_model_trainer.py',
                'description': 'Train ensemble models with sparse data optimization',
                'timeout': None,  # No timeout - let it run as long as needed
                'retry_count': 1,
                'critical': True,
                'depends_on': ['temporal_splitting']
            },
            {
                'name': 'forecasting',
                'script': 'unemployment_forecaster_fixed.py',
                'description': 'Generate production forecasts with validation',
                'timeout': None,  # No timeout - let it run as long as needed
                'retry_count': 2,
                'critical': True,
                'depends_on': ['model_training']
            }
        ]
        
        self.logger.info(f"Initialized production pipeline: {self.pipeline_id}")
    
    def setup_logging(self):
        """Setup structured logging for production monitoring"""
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Create logger
        self.logger = logging.getLogger('ProductionPipeline')
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler with rotation
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """Load enhanced configuration"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            self.logger.info(f"Loaded configuration from {config_file}")
            return config
        except Exception as e:
            self.logger.warning(f"Could not load config {config_file}: {e}")
            return {}
    
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
    
    def monitor_system_resources(self) -> Dict[str, float]:
        """Monitor system resource usage"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent(interval=1)
            
            # System-wide metrics
            system_memory = psutil.virtual_memory()
            system_cpu = psutil.cpu_percent(interval=1)
            
            return {
                'process_memory_mb': memory_info.rss / 1024 / 1024,
                'process_cpu_percent': cpu_percent,
                'system_memory_percent': system_memory.percent,
                'system_cpu_percent': system_cpu,
                'available_memory_gb': system_memory.available / 1024 / 1024 / 1024
            }
        except Exception as e:
            self.logger.warning(f"Could not collect system metrics: {e}")
            return {}
    
    def run_task_with_monitoring(self, task_config: Dict[str, Any]) -> TaskResult:
        """Execute task with comprehensive monitoring and retry logic"""
        task_name = task_config['name']
        script_name = task_config['script']
        timeout = task_config.get('timeout', 600)  # Default 600s, but can be None for no timeout
        max_retries = task_config.get('retry_count', 1)
        
        self.logger.info(f"Starting task: {task_name} ({script_name})")
        
        script_path = self.base_dir / script_name
        if not script_path.exists():
            error_msg = f"Script not found: {script_path}"
            self.logger.error(error_msg)
            return TaskResult(
                task_name=task_name,
                status=TaskStatus.FAILED,
                start_time=datetime.now(),
                error_details=error_msg
            )
        
        # Attempt task execution with retry logic
        for attempt in range(max_retries + 1):
            if attempt > 0:
                self.logger.info(f"Retry attempt {attempt} for {task_name}")
            
            start_time = datetime.now()
            start_resources = self.monitor_system_resources()
            
            try:
                # Execute with timeout
                result = subprocess.run(
                    [sys.executable, str(script_path)], 
                    cwd=self.base_dir, 
                    capture_output=True, 
                    text=True,
                    timeout=timeout
                )
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                end_resources = self.monitor_system_resources()
                
                # Calculate resource usage
                memory_usage = end_resources.get('process_memory_mb', 0) - start_resources.get('process_memory_mb', 0)
                cpu_usage = end_resources.get('process_cpu_percent', 0)
                
                task_result = TaskResult(
                    task_name=task_name,
                    status=TaskStatus.COMPLETED if result.returncode == 0 else TaskStatus.FAILED,
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=duration,
                    exit_code=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    memory_usage_mb=memory_usage,
                    cpu_usage_percent=cpu_usage
                )
                
                if result.returncode == 0:
                    self.logger.info(f"Task {task_name} completed successfully in {duration:.2f}s")
                    return task_result
                else:
                    self.logger.error(f"Task {task_name} failed (attempt {attempt + 1}): {result.stderr[:200]}")
                    if attempt == max_retries:
                        task_result.error_details = f"Task failed after {max_retries + 1} attempts"
                        return task_result
                    else:
                        time.sleep(5 * (attempt + 1))  # Exponential backoff
                        
            except subprocess.TimeoutExpired:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                error_msg = f"Task {task_name} timed out after {timeout}s" if timeout else f"Task {task_name} timed out"
                self.logger.error(error_msg)
                
                if attempt == max_retries:
                    return TaskResult(
                        task_name=task_name,
                        status=TaskStatus.FAILED,
                        start_time=start_time,
                        end_time=end_time,
                        duration_seconds=duration,
                        error_details=error_msg
                    )
                else:
                    time.sleep(10 * (attempt + 1))  # Longer backoff for timeouts
                    
            except Exception as e:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                error_msg = f"Unexpected error in {task_name}: {str(e)}"
                self.logger.error(error_msg)
                
                return TaskResult(
                    task_name=task_name,
                    status=TaskStatus.FAILED,
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=duration,
                    error_details=error_msg
                )
        
        # Should never reach here
        return TaskResult(
            task_name=task_name,
            status=TaskStatus.FAILED,
            start_time=start_time,
            error_details="Unexpected execution path"
        )
    
    def check_task_dependencies(self, task_config: Dict[str, Any]) -> bool:
        """Check if task dependencies are satisfied"""
        dependencies = task_config.get('depends_on', [])
        if not dependencies:
            return True
        
        completed_tasks = {result.task_name for result in self.task_results 
                          if result.status == TaskStatus.COMPLETED}
        
        for dep in dependencies:
            # Map dependency names to task names
            dep_task_name = dep
            if dep_task_name not in completed_tasks:
                self.logger.warning(f"Dependency {dep_task_name} not completed for task {task_config['name']}")
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
        if not models_dir.exists() or not (models_dir / 'unemployment_forecasts.json').exists():
            print("ERROR: Forecasts not found")
            return False
            
        print("All outputs verified successfully")
        return True
    
    def save_pipeline_metrics(self):
        """Save comprehensive pipeline metrics for monitoring"""
        try:
            metrics_dir = self.base_dir / "metrics"
            metrics_dir.mkdir(exist_ok=True)
            
            # Update pipeline metrics
            self.pipeline_metrics.end_time = datetime.now()
            self.pipeline_metrics.total_duration_seconds = (
                self.pipeline_metrics.end_time - self.pipeline_metrics.start_time
            ).total_seconds()
            
            # Count task statuses
            for result in self.task_results:
                if result.status == TaskStatus.COMPLETED:
                    self.pipeline_metrics.tasks_completed += 1
                elif result.status == TaskStatus.FAILED:
                    self.pipeline_metrics.tasks_failed += 1
                elif result.status == TaskStatus.SKIPPED:
                    self.pipeline_metrics.tasks_skipped += 1
            
            # Save detailed metrics
            metrics_file = metrics_dir / f"{self.pipeline_id}_metrics.json"
            metrics_data = {
                'pipeline_metrics': self.pipeline_metrics.to_dict(),
                'task_results': [result.to_dict() for result in self.task_results]
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            self.logger.info(f"Pipeline metrics saved: {metrics_file}")
            
            # Create summary for monitoring dashboards
            summary_file = metrics_dir / "latest_pipeline_summary.json"
            summary = {
                'pipeline_id': self.pipeline_id,
                'status': 'SUCCESS' if self.pipeline_metrics.tasks_failed == 0 else 'FAILED',
                'duration_minutes': self.pipeline_metrics.total_duration_seconds / 60,
                'tasks_completed': self.pipeline_metrics.tasks_completed,
                'tasks_failed': self.pipeline_metrics.tasks_failed,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save pipeline metrics: {e}")

    def run_production_pipeline(self):
        """Execute production-grade MLOps pipeline with 2024 best practices"""
        self.logger.info("="*80)
        self.logger.info("PRODUCTION MLOps PIPELINE EXECUTION")
        self.logger.info("="*80)
        
        try:
            # Step 1: Create backup
            self.logger.info("Creating backup of existing models and data...")
            backup_path = self.create_backup()
            
            # Step 2: Execute pipeline tasks with dependency resolution
            self.logger.info("Executing pipeline tasks with dependency management...")
            
            remaining_tasks = self.pipeline_tasks.copy()
            execution_round = 1
            max_rounds = 10  # Prevent infinite loops
            
            while remaining_tasks and execution_round <= max_rounds:
                self.logger.info(f"Pipeline execution round {execution_round}")
                tasks_executed_this_round = []
                
                for task_config in remaining_tasks.copy():
                    # Check dependencies
                    if self.check_task_dependencies(task_config):
                        self.logger.info(f"Dependencies satisfied for {task_config['name']}")
                        
                        # Execute task
                        task_result = self.run_task_with_monitoring(task_config)
                        self.task_results.append(task_result)
                        
                        # Check if task is critical and failed
                        if task_result.status == TaskStatus.FAILED and task_config.get('critical', False):
                            raise Exception(f"Critical task failed: {task_config['name']}")
                        
                        tasks_executed_this_round.append(task_config)
                        remaining_tasks.remove(task_config)
                    else:
                        self.logger.info(f"Dependencies not met for {task_config['name']}, skipping this round")
                
                if not tasks_executed_this_round:
                    # No tasks executed, check for dependency deadlock
                    unmet_deps = []
                    for task in remaining_tasks:
                        deps = task.get('depends_on', [])
                        unmet_deps.extend([dep for dep in deps if dep not in 
                                         [r.task_name for r in self.task_results if r.status == TaskStatus.COMPLETED]])
                    
                    if unmet_deps:
                        raise Exception(f"Dependency deadlock detected. Unmet dependencies: {set(unmet_deps)}")
                    else:
                        break  # No more tasks to execute
                
                execution_round += 1
            
            # Step 3: Verify pipeline outputs
            self.logger.info("Verifying pipeline outputs...")
            if not self.verify_output():
                raise Exception("Output verification failed")
            
            # Step 4: Save comprehensive metrics
            self.save_pipeline_metrics()
            
            # Step 5: Cleanup old backups
            self.cleanup_old_backups(keep_count=10)
            
            # Success summary
            completed_tasks = sum(1 for r in self.task_results if r.status == TaskStatus.COMPLETED)
            failed_tasks = sum(1 for r in self.task_results if r.status == TaskStatus.FAILED)
            total_duration = (datetime.now() - self.pipeline_metrics.start_time).total_seconds()
            
            self.logger.info("="*80)
            self.logger.info("PIPELINE EXECUTION COMPLETE")
            self.logger.info(f"Pipeline ID: {self.pipeline_id}")
            self.logger.info(f"Total Duration: {total_duration/60:.2f} minutes")
            self.logger.info(f"Tasks Completed: {completed_tasks}")
            self.logger.info(f"Tasks Failed: {failed_tasks}")
            self.logger.info(f"Success Rate: {completed_tasks/(completed_tasks+failed_tasks)*100:.1f}%")
            self.logger.info("="*80)
            
            return failed_tasks == 0
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            self.logger.error(f"Backup available for rollback: {backup_path}")
            
            # Save failure metrics
            try:
                self.save_pipeline_metrics()
            except:
                pass
            
            return False

def main():
    """Main entry point for production MLOps pipeline"""
    orchestrator = ProductionOrchestrator()
    success = orchestrator.run_production_pipeline()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()