#!/usr/bin/env python3
"""
Model Performance & Demographic Comparison Dashboard
Quick summary tool for reviewing forecasting accuracy and demographic analysis
"""

import json
from pathlib import Path
from datetime import datetime

def load_json_file(filepath):
    """Load JSON file with error handling"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        print(f"Error reading {filepath}: {e}")
        return None

def display_mae_summary():
    """Display core forecasting MAE performance"""
    print("=" * 70)
    print("CORE FORECASTING MODEL PERFORMANCE (MAE Summary)")
    print("=" * 70)
    
    # Load training summary
    training_summary = load_json_file("models/training_summary.json")
    if not training_summary:
        print("[ERROR] Could not load training summary")
        return
    
    # Load detailed evaluation
    evaluation_report = load_json_file("models/model_evaluation_report.json")
    if not evaluation_report:
        print("[ERROR] Could not load evaluation report")
        return
    
    print(f"Training Date: {training_summary.get('training_date', 'Unknown')}")
    print(f"Dataset: {training_summary.get('dataset_info', {}).get('train_records', '?')} train, "
          f"{training_summary.get('dataset_info', {}).get('validation_records', '?')} validation, "
          f"{training_summary.get('dataset_info', {}).get('test_records', '?')} test records")
    print()
    
    # Display best models summary
    print("BEST MODEL PERFORMANCE (Validation MAE):")
    print("-" * 50)
    best_models = training_summary.get('best_models_by_region', {})
    
    for region, model_info in best_models.items():
        model_name = model_info.get('best_model', 'Unknown')
        mae = model_info.get('validation_mae', 0)
        print(f"  {region:12} | {model_name:15} | {mae:.3f} MAE")
    
    print()
    
    # Display all model comparison
    print("ALL MODELS COMPARISON (Validation MAE):")
    print("-" * 50)
    regions = ['Auckland', 'Wellington', 'Canterbury']
    models = ['arima', 'random_forest', 'gradient_boosting']
    
    # Header
    print(f"{'Region':12} | {'ARIMA':8} | {'Rand.Forest':10} | {'Grad.Boost':10}")
    print("-" * 50)
    
    for region in regions:
        mae_values = []
        for model in models:
            if model in evaluation_report:
                region_data = evaluation_report[model].get(region, {})
                mae = region_data.get('validation_mae', 0)
                mae_values.append(f"{mae:.3f}")
            else:
                mae_values.append("N/A")
        
        print(f"{region:12} | {mae_values[0]:8} | {mae_values[1]:10} | {mae_values[2]:10}")
    
    # Test set performance
    print("\nTEST SET PERFORMANCE (Final Validation):")
    print("-" * 50)
    test_results = evaluation_report.get('test_results', {})
    
    print(f"{'Region':12} | {'ARIMA':8} | {'Rand.Forest':10} | {'Grad.Boost':10}")
    print("-" * 50)
    
    for region in regions:
        if region in test_results:
            arima_mae = test_results[region].get('arima', {}).get('mae', 0)
            rf_mae = test_results[region].get('random_forest', {}).get('mae', 0)
            gb_mae = test_results[region].get('gradient_boosting', {}).get('mae', 0)
            print(f"{region:12} | {arima_mae:.3f}    | {rf_mae:.3f}     | {gb_mae:.3f}")

def display_demographic_comparison():
    """Display demographic comparison analysis"""
    print("\n" + "=" * 70)
    print("DEMOGRAPHIC COMPARISON ANALYSIS")
    print("=" * 70)
    
    # Load demographic comparison report
    demo_report = load_json_file("models/demographic_comparison_report.json")
    if not demo_report:
        print("[ERROR] Could not load demographic comparison report")
        return
    
    # Executive summary
    exec_summary = demo_report.get('executive_summary', {})
    print(f"Analysis Date: {demo_report.get('generation_date', 'Unknown')}")
    print(f"Client Requirement: {exec_summary.get('client_requirement_status', 'Unknown')}")
    print(f"Demographics Analyzed: {len(exec_summary.get('demographics_compared', []))}")
    print()
    
    # Demographics overview
    print("DEMOGRAPHIC UNEMPLOYMENT PATTERNS:")
    print("-" * 50)
    detailed_findings = demo_report.get('detailed_findings', {})
    demo_stats = detailed_findings.get('demographic_statistics', {})
    volatility_analysis = detailed_findings.get('volatility_analysis', {})
    
    print(f"{'Demographic':15} | {'Avg Rate':8} | {'Volatility':10} | {'Category':10}")
    print("-" * 50)
    
    for demo_name in ['European', 'Asian', 'Pacific_Peoples', 'Maori']:
        if demo_name in demo_stats:
            # Calculate average rate across regions for this demographic
            demo_data = demo_stats[demo_name]
            rates = []
            volatilities = []
            
            for series_name, stats in demo_data.items():
                if 'mean_rate' in stats:
                    rates.append(stats['mean_rate'])
                
                # Get volatility info
                if demo_name in volatility_analysis and series_name in volatility_analysis[demo_name]:
                    vol_info = volatility_analysis[demo_name][series_name]
                    volatilities.append(vol_info.get('coefficient_variation', 0))
            
            if rates:
                avg_rate = sum(rates) / len(rates)
                avg_volatility = sum(volatilities) / len(volatilities) if volatilities else 0
                
                # Determine volatility category
                if avg_volatility < 0.2:
                    vol_category = "Low"
                elif avg_volatility < 0.5:
                    vol_category = "Moderate"
                else:
                    vol_category = "High"
                
                print(f"{demo_name:15} | {avg_rate:7.1f}% | {avg_volatility:9.3f} | {vol_category:10}")
    
    # Forecasting implications
    print("\nFORECASTING ACCURACY EXPECTATIONS:")
    print("-" * 50)
    forecasting_implications = demo_report.get('forecasting_implications', {})
    accuracy_expectations = forecasting_implications.get('accuracy_expectations', {})
    
    print(f"{'Demographic':15} | {'Expected MAE':12} | {'Confidence':10}")
    print("-" * 50)
    
    for demo, expectations in accuracy_expectations.items():
        mae_range = expectations.get('expected_mae_range', 'Unknown')
        confidence = expectations.get('forecasting_confidence', 'Unknown')
        print(f"{demo:15} | {mae_range:12} | {confidence:10}")
    
    # High volatility warnings
    print("\nHIGH VOLATILITY WARNINGS:")
    print("-" * 50)
    volatility_warnings = demo_report.get('dashboard_recommendations', {}).get('volatility_warnings', [])
    
    if volatility_warnings:
        for warning in volatility_warnings:
            series = warning.get('series', 'Unknown')
            demo = warning.get('demographic', 'Unknown')
            level = warning.get('volatility_level', 'Unknown')
            cv = warning.get('coefficient_variation', 0)
            print(f"  [WARNING] {series} ({demo}) - {level} volatility (CV: {cv:.3f})")
    else:
        print("  [OK] No high volatility warnings")
    
    # Requirements compliance
    requirements_compliance = detailed_findings.get('requirements_compliance', {})
    comparison_enabled = requirements_compliance.get('demographic_comparison_enabled', False)
    client_fulfilled = requirements_compliance.get('client_requirement_fulfilled', 'Unknown')
    
    print("\nREQUIREMENTS.MD COMPLIANCE:")
    print("-" * 50)
    print(f"  Demographic Comparison: {'[ENABLED]' if comparison_enabled else '[DISABLED]'}")
    print(f"  Client Requirement: {client_fulfilled}")
    print(f"  Analysis Scope: {requirements_compliance.get('analysis_scope', 'Unknown')}")

def display_system_summary():
    """Display overall system status"""
    print("\n" + "=" * 70)
    print("SYSTEM SUMMARY")
    print("=" * 70)
    
    # Check file availability
    files_to_check = [
        "models/training_summary.json",
        "models/model_evaluation_report.json", 
        "models/demographic_comparison_report.json",
        "models/fixed_unemployment_forecasts.json"
    ]
    
    print("SYSTEM FILES STATUS:")
    print("-" * 30)
    all_files_present = True
    
    for file_path in files_to_check:
        if Path(file_path).exists():
            file_time = datetime.fromtimestamp(Path(file_path).stat().st_mtime)
            hours_old = (datetime.now() - file_time).total_seconds() / 3600
            print(f"  [OK] {Path(file_path).name:30} ({hours_old:.1f}h old)")
        else:
            print(f"  [MISSING] {Path(file_path).name:30}")
            all_files_present = False
    
    print("\nSYSTEM READINESS:")
    print("-" * 20)
    if all_files_present:
        print("  [OK] Production Forecasting: READY")
        print("  [OK] Demographic Analysis: READY")
        print("  [OK] MBIE Presentation: READY")
    else:
        print("  [WARNING] System incomplete - missing files detected")
    
    print("\nRECOMMEND ACTIONS:")
    print("-" * 20)
    print("  1. Review MAE performance above (target: <1.0 for stable demographics)")
    print("  2. Check demographic comparison data for client presentation")
    print("  3. Run orchestrator (py quarterly_update_orchestrator.py) if files missing")
    print("  4. Use demographic analysis for dashboard comparison features")

def main():
    """Main dashboard display function"""
    print("NZ UNEMPLOYMENT FORECASTING - MODEL PERFORMANCE DASHBOARD")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        display_mae_summary()
        display_demographic_comparison()  
        display_system_summary()
        
        print("\n" + "=" * 70)
        print("DASHBOARD COMPLETE")
        print("=" * 70)
        print("Use this data for:")
        print("  • Model performance evaluation")
        print("  • Demographic comparison insights")
        print("  • MBIE presentation preparation")
        print("  • Dashboard UI planning")
        
    except Exception as e:
        print(f"\n[ERROR] Dashboard generation failed: {e}")
        print("Check that all model files exist and are valid JSON")

if __name__ == "__main__":
    main()