#!/usr/bin/env python3
"""
MBIE MLOps Pipeline GUI - Professional Interface
Modern desktop application for unemployment forecasting pipeline orchestration
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog, font
import subprocess
import threading
import json
import os
import sys
from datetime import datetime
import queue
import time
import webbrowser

class MLOpsGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.setup_window()
        self.setup_styles()
        self.create_widgets()
        self.setup_layout()
        
        # Process management
        self.current_process = None
        self.log_queue = queue.Queue()
        self.is_running = False
        
        # Load configuration
        self.load_config()
        
    def setup_window(self):
        self.root.title("MBIE MLOps Pipeline")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # GitHub-style dark theme
        self.root.configure(bg='#0d1117')
        
        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (1400 // 2)
        y = (self.root.winfo_screenheight() // 2) - (900 // 2)
        self.root.geometry(f"1400x900+{x}+{y}")
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # GitHub-style professional colors
        self.colors = {
            'bg_primary': '#0d1117',       # GitHub dark background
            'bg_secondary': '#161b22',     # GitHub secondary background
            'bg_tertiary': '#21262d',      # GitHub tertiary background
            'border': '#30363d',           # GitHub border
            'border_muted': '#21262d',     # Muted border
            'accent_blue': '#58a6ff',      # GitHub blue
            'accent_green': '#3fb950',     # GitHub green
            'accent_red': '#f85149',       # GitHub red
            'accent_orange': '#d29922',    # GitHub orange
            'text_primary': '#f0f6fc',     # Primary text
            'text_secondary': '#8b949e',   # Secondary text
            'text_muted': '#6e7681',       # Muted text
            'hover': '#262c36'            # Hover background
        }
        
        # GitHub-style fonts (using supported font weights)
        self.fonts = {
            'title': ('Segoe UI', 20, 'normal'),
            'header': ('Segoe UI', 14, 'bold'),
            'button': ('Segoe UI', 11, 'normal'),
            'text': ('Segoe UI', 10, 'normal'),
            'mono': ('Consolas', 11, 'normal')
        }
        
        # Configure GitHub-style themes
        style.configure('Title.TLabel', 
                       font=self.fonts['title'],
                       foreground=self.colors['text_primary'],
                       background=self.colors['bg_primary'])
        
        style.configure('Header.TLabel', 
                       font=self.fonts['header'],
                       foreground=self.colors['text_primary'],
                       background=self.colors['bg_secondary'])
        
        style.configure('Primary.TButton',
                       font=self.fonts['button'],
                       background=self.colors['accent_blue'],
                       foreground='#ffffff',
                       borderwidth=1,
                       relief='flat',
                       focuscolor='none')
        
        style.configure('Success.TButton',
                       font=self.fonts['button'],
                       background=self.colors['accent_green'],
                       foreground='#ffffff',
                       borderwidth=1,
                       relief='flat')
        
        style.configure('Warning.TButton',
                       font=self.fonts['button'],
                       background=self.colors['accent_orange'],
                       foreground='#ffffff',
                       borderwidth=1,
                       relief='flat')
        
        style.configure('Danger.TButton',
                       font=self.fonts['button'],
                       background=self.colors['accent_red'],
                       foreground='#ffffff',
                       borderwidth=1,
                       relief='flat')
        
        # Progress bar styling
        style.configure('GitHub.Horizontal.TProgressbar',
                       background=self.colors['accent_blue'],
                       troughcolor=self.colors['bg_tertiary'],
                       borderwidth=1,
                       relief='flat')
                       
    def create_widgets(self):
        # Main container with GitHub styling
        self.main_frame = tk.Frame(self.root, bg=self.colors['bg_primary'], padx=24, pady=24)
        
        # Clean header
        self.create_header()
        
        # Professional control panel
        self.create_control_panel()
        
        # Status dashboard
        self.create_status_panel()
        
        # Log panel
        self.create_log_panel()
        
        # Progress panel
        self.create_progress_panel()
        
    def create_header(self):
        """Create clean professional header"""
        header_frame = tk.Frame(self.main_frame, bg=self.colors['bg_primary'])
        
        # Header container with GitHub-style border
        header_container = tk.Frame(header_frame, 
                                   bg=self.colors['bg_secondary'], 
                                   highlightbackground=self.colors['border'],
                                   highlightthickness=1)
        header_container.pack(fill='x', pady=(0, 24), ipady=20, ipadx=20)
        
        # Title
        title_label = tk.Label(header_container, 
                              text="MBIE MLOps Pipeline",
                              font=self.fonts['title'],
                              fg=self.colors['text_primary'],
                              bg=self.colors['bg_secondary'])
        title_label.pack(anchor='w')
        
        # Subtitle
        subtitle_label = tk.Label(header_container,
                                 text="Unemployment Forecasting System",
                                 font=self.fonts['text'],
                                 fg=self.colors['text_secondary'],
                                 bg=self.colors['bg_secondary'])
        subtitle_label.pack(anchor='w', pady=(4, 0))
        
        self.header_frame = header_frame
        
    def create_control_panel(self):
        """Create clean GitHub-style control panel"""
        # Main control frame
        control_frame = tk.Frame(self.main_frame, 
                                bg=self.colors['bg_secondary'],
                                highlightbackground=self.colors['border'],
                                highlightthickness=1)
        
        # Header
        header_frame = tk.Frame(control_frame, bg=self.colors['bg_secondary'])
        header_frame.pack(fill='x', padx=20, pady=(16, 12))
        
        header_label = tk.Label(header_frame, text="Pipeline Control",
                               font=self.fonts['header'], 
                               fg=self.colors['text_primary'],
                               bg=self.colors['bg_secondary'])
        header_label.pack(anchor='w')
        
        # Pipeline steps with clean styling
        steps_frame = tk.Frame(control_frame, bg=self.colors['bg_secondary'])
        steps_frame.pack(fill='x', padx=20, pady=(0, 16))
        
        # Individual step buttons
        self.step_buttons = {}
        steps = [
            ("1. Data Cleaner", "data_cleaner.py", "Prepare and clean raw datasets"),
            ("2. Time Series Aligner", "time_series_aligner_simplified.py", "Align temporal data structures"),
            ("3. Temporal Splitter", "temporal_data_splitter.py", "Split data for ML training"),
            ("4. Model Trainer", "unemployment_model_trainer.py", "Train Random Forest models"),
            ("5. Forecaster", "unemployment_forecaster_fixed.py", "Generate unemployment forecasts")
        ]
        
        for i, (name, script, desc) in enumerate(steps):
            # Step container with GitHub-style card
            step_container = tk.Frame(steps_frame, 
                                     bg=self.colors['bg_tertiary'],
                                     highlightbackground=self.colors['border_muted'],
                                     highlightthickness=1)
            step_container.pack(fill='x', pady=3)
            
            step_frame = tk.Frame(step_container, bg=self.colors['bg_tertiary'])
            step_frame.pack(fill='x', padx=16, pady=12)
            
            # Clean button
            btn = tk.Button(step_frame, text=name, width=20,
                           command=lambda s=script: self.run_script(s),
                           font=self.fonts['button'],
                           bg=self.colors['accent_blue'],
                           fg='#ffffff',
                           activebackground='#316dca',
                           border=0,
                           relief='flat',
                           cursor='hand2')
            btn.pack(side='left', padx=(0, 16))
            
            # Description
            desc_label = tk.Label(step_frame, text=desc, 
                                 font=self.fonts['text'],
                                 fg=self.colors['text_secondary'],
                                 bg=self.colors['bg_tertiary'])
            desc_label.pack(side='left', anchor='w')
            
            self.step_buttons[script] = btn
        
        # Master control panel
        master_frame = tk.Frame(control_frame, bg=self.colors['bg_secondary'])
        master_frame.pack(fill='x', padx=20, pady=(16, 20))
        
        # Main control buttons
        tk.Button(master_frame, text="Run Full Pipeline",
                 command=self.run_full_pipeline,
                 font=self.fonts['button'],
                 bg=self.colors['accent_green'],
                 fg='#ffffff',
                 activebackground='#2ea043',
                 border=0,
                 relief='flat',
                 cursor='hand2',
                 width=18).pack(side='left', padx=(0, 12))
        
        tk.Button(master_frame, text="Stop Pipeline",
                 command=self.stop_pipeline,
                 font=self.fonts['button'],
                 bg=self.colors['accent_red'],
                 fg='#ffffff',
                 activebackground='#da3633',
                 border=0,
                 relief='flat',
                 cursor='hand2',
                 width=14).pack(side='left', padx=(0, 12))
        
        tk.Button(master_frame, text="View Results",
                 command=self.view_results,
                 font=self.fonts['button'],
                 bg=self.colors['accent_orange'],
                 fg='#ffffff',
                 activebackground='#bf8700',
                 border=0,
                 relief='flat',
                 cursor='hand2',
                 width=14).pack(side='left')
        
        self.control_frame = control_frame
        
    def create_status_panel(self):
        """Create GitHub-style status panel"""
        status_frame = tk.Frame(self.main_frame, 
                               bg=self.colors['bg_secondary'],
                               highlightbackground=self.colors['border'],
                               highlightthickness=1)
        
        # Status header
        header_frame = tk.Frame(status_frame, bg=self.colors['bg_secondary'])
        header_frame.pack(fill='x', padx=20, pady=(16, 12))
        
        header_label = tk.Label(header_frame, text="System Status",
                               font=self.fonts['header'], 
                               fg=self.colors['text_primary'],
                               bg=self.colors['bg_secondary'])
        header_label.pack(anchor='w')
        
        # Status grid with clean styling
        status_grid = tk.Frame(status_frame, bg=self.colors['bg_secondary'])
        status_grid.pack(fill='x', padx=20, pady=(0, 16))
        
        # Clean status indicators in a grid
        status_items = [
            ("Current Status:", "status_label", "Ready", self.colors['accent_green']),
            ("Last Run:", "last_run_label", "Never", self.colors['text_secondary']),
            ("Models Trained:", "models_label", "0", self.colors['accent_blue']),
            ("Data Points:", "data_points_label", "0", self.colors['text_secondary'])
        ]
        
        for i, (label_text, attr_name, default_text, color) in enumerate(status_items):
            # Create status row with GitHub card styling
            row_frame = tk.Frame(status_grid, 
                                bg=self.colors['bg_tertiary'],
                                highlightbackground=self.colors['border_muted'],
                                highlightthickness=1)
            row_frame.pack(fill='x', pady=2)
            
            row_inner = tk.Frame(row_frame, bg=self.colors['bg_tertiary'])
            row_inner.pack(fill='x', padx=16, pady=8)
            
            # Status label
            label = tk.Label(row_inner, text=label_text, 
                            font=('Segoe UI', 10, 'bold'),
                            fg=self.colors['text_primary'],
                            bg=self.colors['bg_tertiary'])
            label.pack(side='left')
            
            # Status value
            status_label = tk.Label(row_inner, text=default_text,
                                   font=self.fonts['text'],
                                   fg=color,
                                   bg=self.colors['bg_tertiary'])
            status_label.pack(side='right')
            setattr(self, attr_name, status_label)
        
        self.status_frame = status_frame
        
    def create_log_panel(self):
        """Create GitHub-style log panel"""
        log_frame = tk.Frame(self.main_frame, 
                            bg=self.colors['bg_secondary'],
                            highlightbackground=self.colors['border'],
                            highlightthickness=1)
        
        # Header
        header_frame = tk.Frame(log_frame, bg=self.colors['bg_secondary'])
        header_frame.pack(fill='x', padx=20, pady=(16, 12))
        
        header_label = tk.Label(header_frame, text="Pipeline Logs",
                               font=self.fonts['header'], 
                               fg=self.colors['text_primary'],
                               bg=self.colors['bg_secondary'])
        header_label.pack(side='left')
        
        # Clean log display with GitHub styling
        log_container = tk.Frame(log_frame, bg=self.colors['bg_secondary'])
        log_container.pack(fill='both', expand=True, padx=20, pady=(0, 12))
        
        self.log_text = scrolledtext.ScrolledText(log_container, 
                                                 height=15, 
                                                 font=self.fonts['mono'],
                                                 background=self.colors['bg_tertiary'],
                                                 foreground=self.colors['text_primary'],
                                                 insertbackground=self.colors['text_primary'],
                                                 selectbackground=self.colors['hover'],
                                                 selectforeground=self.colors['text_primary'],
                                                 relief='flat',
                                                 borderwidth=1,
                                                 highlightcolor=self.colors['border'],
                                                 highlightthickness=1)
        self.log_text.pack(fill='both', expand=True)
        
        # Clean control buttons
        log_controls = tk.Frame(log_frame, bg=self.colors['bg_secondary'])
        log_controls.pack(fill='x', padx=20, pady=(0, 16))
        
        tk.Button(log_controls, text="Clear Logs",
                 command=self.clear_logs,
                 font=self.fonts['button'],
                 bg=self.colors['bg_tertiary'],
                 fg=self.colors['text_primary'],
                 activebackground=self.colors['hover'],
                 border=1,
                 relief='flat',
                 cursor='hand2').pack(side='left', padx=(0, 8))
        
        tk.Button(log_controls, text="Save Logs",
                 command=self.save_logs,
                 font=self.fonts['button'],
                 bg=self.colors['bg_tertiary'],
                 fg=self.colors['text_primary'],
                 activebackground=self.colors['hover'],
                 border=1,
                 relief='flat',
                 cursor='hand2').pack(side='left', padx=(0, 8))
        
        tk.Button(log_controls, text="Auto-scroll",
                 command=self.toggle_autoscroll,
                 font=self.fonts['button'],
                 bg=self.colors['bg_tertiary'],
                 fg=self.colors['text_primary'],
                 activebackground=self.colors['hover'],
                 border=1,
                 relief='flat',
                 cursor='hand2').pack(side='right')
        
        self.log_frame = log_frame
        self.autoscroll = True
        
    def create_progress_panel(self):
        """Create clean GitHub-style progress panel"""
        progress_frame = tk.Frame(self.main_frame, bg=self.colors['bg_primary'])
        
        # Progress container with subtle styling
        progress_container = tk.Frame(progress_frame, 
                                     bg=self.colors['bg_secondary'],
                                     highlightbackground=self.colors['border'],
                                     highlightthickness=1)
        progress_container.pack(fill='x', pady=(0, 0))
        
        # Progress bar with GitHub styling
        progress_inner = tk.Frame(progress_container, bg=self.colors['bg_secondary'])
        progress_inner.pack(fill='x', padx=20, pady=12)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_inner, 
                                          variable=self.progress_var,
                                          maximum=100,
                                          style='GitHub.Horizontal.TProgressbar')
        self.progress_bar.pack(fill='x', pady=(0, 8))
        
        # Clean progress label
        self.progress_label = tk.Label(progress_inner, text="Ready to start pipeline",
                                      font=self.fonts['text'],
                                      fg=self.colors['text_secondary'],
                                      bg=self.colors['bg_secondary'])
        self.progress_label.pack(anchor='w')
        
        self.progress_frame = progress_frame
        
    def setup_layout(self):
        """Setup ultra-modern gaming layout"""
        self.main_frame.pack(fill='both', expand=True)
        
        self.header_frame.pack(fill='x', pady=(0, 15))
        self.control_frame.pack(fill='x', pady=(0, 15))
        self.status_frame.pack(fill='x', pady=(0, 15))
        self.log_frame.pack(fill='both', expand=True, pady=(0, 15))
        self.progress_frame.pack(fill='x')
        
    def load_config(self):
        """Load configuration and update status"""
        try:
            if os.path.exists('simple_config.json'):
                with open('simple_config.json', 'r') as f:
                    self.config = json.load(f)
                self.log_message("Configuration loaded successfully", "SUCCESS")
            else:
                self.log_message("Configuration file not found", "WARNING")
                self.config = {}
        except Exception as e:
            self.log_message(f"Error loading configuration: {e}", "ERROR")
            self.config = {}
            
        self.update_status()
        
    def update_status(self):
        """Update status panel with current information"""
        try:
            # Count model files
            models_dir = "models"
            if os.path.exists(models_dir):
                model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
                self.models_label.config(text=str(len(model_files)))
            
            # Check integration metrics
            if os.path.exists('data_cleaned/integration_metrics.json'):
                with open('data_cleaned/integration_metrics.json', 'r') as f:
                    metrics = json.load(f)
                    self.data_points_label.config(text=f"{metrics.get('total_periods', 0):,}")
            
        except Exception as e:
            self.log_message(f"Error updating status: {e}", "WARNING")
            
    def log_message(self, message, level="INFO"):
        """Add professional message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # GitHub-style color scheme
        colors = {
            "INFO": self.colors['text_primary'],
            "SUCCESS": self.colors['accent_green'],
            "WARNING": self.colors['accent_orange'],
            "ERROR": self.colors['accent_red'],
            "DEBUG": self.colors['text_secondary']
        }
        
        formatted_msg = f"[{timestamp}] [{level}] {message}\n"
        
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, formatted_msg)
        
        # Apply colors to the message
        last_line_start = self.log_text.index("end-2c linestart")
        last_line_end = self.log_text.index("end-1c")
        self.log_text.tag_add(level, last_line_start, last_line_end)
        self.log_text.tag_config(level, foreground=colors.get(level, self.colors['text_primary']))
        
        self.log_text.configure(state='disabled')
        
        if self.autoscroll:
            self.log_text.see(tk.END)
            
    def run_script(self, script_name):
        """Run individual pipeline script"""
        if self.is_running:
            messagebox.showwarning("Warning", "Pipeline is already running!")
            return
            
        self.is_running = True
        self.update_button_states(False)
        self.status_label.config(text="Running", fg=self.colors['accent_orange'])
        self.progress_label.config(text=f"Running {script_name}...")
        
        def run_in_thread():
            try:
                self.log_message(f"Starting {script_name}...", "INFO")
                
                process = subprocess.Popen(
                    [sys.executable, script_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                self.current_process = process
                
                # Read output line by line
                for line in process.stdout:
                    line = line.strip()
                    if line:
                        self.root.after(0, lambda l=line: self.log_message(l, "DEBUG"))
                
                process.wait()
                
                if process.returncode == 0:
                    self.root.after(0, lambda: self.log_message(f"[SUCCESS] {script_name} MISSION COMPLETE!", "SUCCESS"))
                    self.root.after(0, lambda: self.status_label.config(text="MISSION_COMPLETE", fg=self.colors['accent_green']))
                else:
                    self.root.after(0, lambda: self.log_message(f"[CRITICAL] {script_name} EXECUTION FAILED - CODE {process.returncode}", "ERROR"))
                    self.root.after(0, lambda: self.status_label.config(text="SYSTEM_ERROR", fg=self.colors['accent_red']))
                    
            except Exception as e:
                self.root.after(0, lambda: self.log_message(f"[CRITICAL] Runtime error in {script_name}: {e}", "ERROR"))
                self.root.after(0, lambda: self.status_label.config(text="CRITICAL_ERROR", fg=self.colors['accent_red']))
            finally:
                self.current_process = None
                self.is_running = False
                self.root.after(0, lambda: self.update_button_states(True))
                self.root.after(0, lambda: self.progress_label.config(text=">> SYSTEM READY FOR NEXT DEPLOYMENT <<"))
                self.root.after(0, self.update_status)
                self.root.after(0, lambda: self.last_run_label.config(text=f"EXECUTED_{datetime.now().strftime('%H%M%S')}", fg=self.colors['accent_green']))
                
        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()
        
    def run_full_pipeline(self):
        """Run complete pipeline sequence"""
        if self.is_running:
            messagebox.showwarning("Warning", "Pipeline is already running!")
            return
            
        scripts = [
            "data_cleaner.py",
            "time_series_aligner_simplified.py", 
            "temporal_data_splitter.py",
            "unemployment_model_trainer.py",
            "unemployment_forecaster_fixed.py"
        ]
        
        self.is_running = True
        self.update_button_states(False)
        self.status_label.config(text="Running Full Pipeline", foreground="orange")
        
        def run_pipeline():
            try:
                self.log_message("=== STARTING FULL PIPELINE ===", "INFO")
                
                for i, script in enumerate(scripts, 1):
                    progress = (i - 1) / len(scripts) * 100
                    self.root.after(0, lambda p=progress: self.progress_var.set(p))
                    self.root.after(0, lambda s=script, i=i: self.progress_label.config(text=f"Step {i}/5: {s}"))
                    
                    self.root.after(0, lambda s=script: self.log_message(f"Starting step {i}/5: {s}", "INFO"))
                    
                    process = subprocess.Popen(
                        [sys.executable, script],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        universal_newlines=True,
                        bufsize=1
                    )
                    
                    self.current_process = process
                    
                    for line in process.stdout:
                        line = line.strip()
                        if line:
                            self.root.after(0, lambda l=line: self.log_message(l, "DEBUG"))
                    
                    process.wait()
                    
                    if process.returncode != 0:
                        self.root.after(0, lambda s=script: self.log_message(f"Pipeline failed at {s}", "ERROR"))
                        self.root.after(0, lambda: self.status_label.config(text="Failed", foreground="red"))
                        return
                    
                    self.root.after(0, lambda s=script: self.log_message(f"Completed: {s}", "SUCCESS"))
                
                self.root.after(0, lambda: self.progress_var.set(100))
                self.root.after(0, lambda: self.log_message("=== PIPELINE COMPLETED SUCCESSFULLY ===", "SUCCESS"))
                self.root.after(0, lambda: self.status_label.config(text="Pipeline Complete", foreground="green"))
                
            except Exception as e:
                self.root.after(0, lambda: self.log_message(f"Pipeline error: {e}", "ERROR"))
                self.root.after(0, lambda: self.status_label.config(text="Error", foreground="red"))
            finally:
                self.current_process = None
                self.is_running = False
                self.root.after(0, lambda: self.update_button_states(True))
                self.root.after(0, lambda: self.progress_label.config(text="Pipeline Complete"))
                self.root.after(0, self.update_status)
                self.root.after(0, lambda: self.last_run_label.config(text=f"EXECUTED_{datetime.now().strftime('%H%M%S')}", fg=self.colors['accent_green']))
                
        thread = threading.Thread(target=run_pipeline, daemon=True)
        thread.start()
        
    def stop_pipeline(self):
        """Stop currently running pipeline"""
        if self.current_process:
            try:
                self.current_process.terminate()
                self.log_message("Pipeline stopped by user", "WARNING")
                self.status_label.config(text="Stopped", foreground="red")
                self.is_running = False
                self.update_button_states(True)
                self.progress_label.config(text="Stopped")
            except Exception as e:
                self.log_message(f"Error stopping pipeline: {e}", "ERROR")
        else:
            messagebox.showinfo("Info", "No pipeline is currently running")
            
    def view_results(self):
        """Open results viewer"""
        try:
            if os.path.exists('unemployment_forecasts_powerbi.csv'):
                # Open results in default application
                if sys.platform.startswith('win'):
                    os.startfile('unemployment_forecasts_powerbi.csv')
                elif sys.platform.startswith('darwin'):
                    subprocess.call(['open', 'unemployment_forecasts_powerbi.csv'])
                else:
                    subprocess.call(['xdg-open', 'unemployment_forecasts_powerbi.csv'])
                self.log_message("Opening results file...", "INFO")
            else:
                messagebox.showwarning("Warning", "No results file found. Run the pipeline first.")
        except Exception as e:
            self.log_message(f"Error opening results: {e}", "ERROR")
            
    def update_button_states(self, enabled):
        """Enable/disable buttons based on running state"""
        state = 'normal' if enabled else 'disabled'
        for button in self.step_buttons.values():
            button.config(state=state)
            
    def clear_logs(self):
        """Clear the log display"""
        self.log_text.configure(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state='disabled')
        
    def save_logs(self):
        """Save logs to file"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".log",
                filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")]
            )
            if filename:
                with open(filename, 'w') as f:
                    f.write(self.log_text.get(1.0, tk.END))
                self.log_message(f"Logs saved to {filename}", "SUCCESS")
        except Exception as e:
            self.log_message(f"Error saving logs: {e}", "ERROR")
            
    def toggle_autoscroll(self):
        """Toggle auto-scroll functionality"""
        self.autoscroll = not self.autoscroll
        status = "enabled" if self.autoscroll else "disabled"
        self.log_message(f"Auto-scroll {status}", "INFO")
        
    def run(self):
        """Start the GUI application"""
        self.log_message("MBIE MLOps Pipeline GUI initialized", "SUCCESS")
        self.log_message("Ready for pipeline operations", "INFO")
        self.root.mainloop()

def main():
    """Main entry point"""
    try:
        # Change to script directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # Create and run GUI
        app = MLOpsGUI()
        app.run()
        
    except Exception as e:
        print(f"Error starting GUI: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()