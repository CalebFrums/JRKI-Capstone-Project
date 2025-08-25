#!/usr/bin/env python3
"""
Comprehensive Data Cleaner - Essential Version
NZ Unemployment Forecasting System - Government Data Processing

Robust data cleaning for complex government CSV structures with minimal configuration.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import re
import warnings
warnings.filterwarnings('ignore')

class GovernmentDataCleaner:
    """Essential data cleaner for NZ government datasets with robust CSV handling"""
    
    def __init__(self, source_dir="raw_datasets", output_dir="data_cleaned", config_file="simple_config.json"):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.audit_log = []
        self.data_quality_metrics = {}
        
        # Load configuration
        self.config = self.load_config(config_file)
        self.ethnic_groups = self.config.get('demographics', {}).get('ethnic_groups', [])
        self.sex_categories = self.config.get('demographics', {}).get('sex_categories', [])
        self.age_groups = self.config.get('demographics', {}).get('age_groups_basic', [])
        self.age_groups_detailed = self.config.get('demographics', {}).get('age_groups_detailed', [])
        self.priority_regions = self.config.get('forecasting', {}).get('target_columns', {}).get('priority_regions', [])
        
    def load_config(self, config_file):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config {config_file}: {e}")
            return {}
        
    def log_action(self, action, message, level="INFO"):
        """Simple logging"""
        log_entry = {"action": action, "message": message, "level": level}
        self.audit_log.append(log_entry)
        
    def detect_header_structure(self, filepath, max_rows=10):
        """Detect optimal header configuration for complex government CSVs including MEI files"""
        best_config = {"header_rows": 0, "columns": 0, "quality": 0}
        
        # Special handling for MEI files with complex 4-5 row headers
        if "MEI" in str(filepath).upper():
            mei_result = self.detect_mei_header_structure(filepath, max_rows)
            # If MEI detection fails, try fallback
            if mei_result.get("quality", 0) > 0:
                return mei_result
            print(f"    MEI detection failed, trying fallback for {filepath.name}")
        
        # Special handling for HLF files with 4-level demographic headers
        if "HLF" in str(filepath).upper():
            return self.detect_hlf_header_structure(filepath, max_rows)
        
        # Special handling for QEM files with 4-level headers (check before ECT)
        if "QEM" in str(filepath).upper():
            # Check if this is a QEM sector file with percentage changes (needs 5-level headers)
            if "percentage change" in str(filepath).lower() or "sector" in str(filepath).lower():
                return self.detect_qem_sector_header_structure(filepath, max_rows)
            else:
                return self.detect_qem_header_structure(filepath, max_rows)
        
        # Special handling for ECT files with 3-level headers
        # Use more specific matching to avoid false positives (e.g., "SECTORS" contains "ECT")
        filepath_upper = str(filepath).upper()
        if filepath_upper.startswith("ECT") or "/ECT " in filepath_upper or "\\ECT " in filepath_upper:
            return self.detect_ect_header_structure(filepath, max_rows)
        
        # Special handling for LCI files (Labour Cost Index)
        if "LCI" in filepath_upper:
            lci_result = self.detect_lci_header_structure(filepath, max_rows)
            # If LCI detection fails, try fallback
            if lci_result.get("quality", 0) > 0:
                return lci_result
            print(f"    LCI detection failed, trying fallback for {filepath.name}")
        
        for header_rows in range(5):  # Try 0-4 header rows
            try:
                if header_rows == 0:
                    df = pd.read_csv(filepath, nrows=max_rows)
                else:
                    df = pd.read_csv(filepath, header=list(range(header_rows)), nrows=max_rows)
                
                if df.empty or len(df.columns) < 2:
                    continue
                    
                # Calculate quality score
                quality = 0
                for col in df.columns:
                    col_str = str(col).strip()
                    if col_str and 'Unnamed' not in col_str and '..' not in col_str:
                        quality += 1
                
                # Bonus for date-like first column
                if len(df.columns) > 0:
                    first_col_sample = str(df.iloc[0, 0]) if len(df) > 0 else ""
                    if any(char in first_col_sample for char in ['Q', '/', '-']) or first_col_sample.isdigit():
                        quality += 2
                
                if quality > best_config["quality"]:
                    best_config = {
                        "header_rows": header_rows,
                        "columns": len(df.columns),
                        "quality": quality,
                        "sample_df": df
                    }
                    
            except Exception:
                continue
                
        return best_config
    
    def detect_mei_header_structure(self, filepath, max_rows=10):
        """Special header detection for MEI files with complex multi-level structure"""
        print(f"  MEI: Detecting complex header structure for {filepath.name}")
        
        # MEI files typically have 4-5 header rows
        # Row 1: Title, Row 2: Category, Row 3: Age groups, Row 4: Regions, Row 5: Measurement type
        try:
            # Try with 4 header rows (most common for MEI)
            df = pd.read_csv(filepath, header=[0, 1, 2, 3, 4], nrows=max_rows)
            
            # Calculate quality based on meaningful column content
            quality = 0
            age_count = 0
            region_count = 0
            
            for col in df.columns:
                if isinstance(col, tuple) and len(col) >= 3:
                    # Check for age groups from config
                    age_text = str(col[2]) if len(col) > 2 else ""
                    for age_group in self.age_groups_detailed:
                        if age_group in age_text:
                            age_count += 1
                            quality += 2
                            break
                    
                    # Check for regions from config
                    region_text = str(col[3]) if len(col) > 3 else ""
                    for region_list in self.config.get('regions', {}).values():
                        if isinstance(region_list, list):
                            for region in region_list:
                                if region.replace('_', ' ').replace('-', ' ') in region_text:
                                    region_count += 1
                                    quality += 1
                                    break
            
            print(f"    Found {age_count} age groups and {region_count} regions")
            
            # If no age groups or regions found, try industry-based detection
            if quality == 0:
                print("    No age/region patterns found, trying industry-based MEI detection")
                return self.detect_mei_industry_header_structure(filepath, max_rows)
            
            return {
                "header_rows": 5,  # Using 5 header rows (0,1,2,3,4)
                "columns": len(df.columns),
                "quality": quality,
                "sample_df": df,
                "is_mei": True
            }
            
        except Exception as e:
            print(f"    Warning: MEI age/region detection failed: {e}")
            # Try industry-based MEI detection
            return self.detect_mei_industry_header_structure(filepath, max_rows)
    
    def detect_mei_industry_header_structure(self, filepath, max_rows=10):
        """Special header detection for MEI files with industry-based structure"""
        filepath_obj = self.source_dir / filepath if isinstance(filepath, str) else filepath
        print(f"  MEI-INDUSTRY: Detecting industry-based header structure for {filepath_obj.name}")
        
        try:
            # Industry MEI files have 4 header rows
            # Row 1: Title, Row 2: Industries, Row 3: Adjustment types, Row 4: Metrics
            df = pd.read_csv(filepath_obj, header=[0, 1, 2, 3], nrows=max_rows)
            
            # Calculate quality based on industry categories and metrics found
            quality = 0
            industry_count = 0
            adjustment_count = 0
            metric_count = 0
            
            # Get MEI high-level industries from config
            mei_high_level = self.config.get('industries', {}).get('mei_high_level', [])
            mei_industries = self.config.get('industries', {}).get('mei_industries', [])
            
            for col in df.columns:
                if isinstance(col, tuple) and len(col) >= 4:
                    # Check for industry categories - MEI Industry files have industries in level 1 or 2
                    industry_text_level1 = str(col[1]) if len(col) > 1 else ""
                    industry_text_level2 = str(col[2]) if len(col) > 2 else ""
                    # Check for metrics in level 1 or 3
                    metric_text = str(col[1]) if len(col) > 1 else ""
                    metric_text_level3 = str(col[3]) if len(col) > 3 else ""
                    # Check for adjustment types in level 2 or 3  
                    adjustment_text_level2 = str(col[2]) if len(col) > 2 else ""
                    adjustment_text_level3 = str(col[3]) if len(col) > 3 else ""
                    
                    # Check against config high-level industries in both level 1 and level 2
                    industry_found = False
                    for industry in mei_high_level:
                        industry_clean = industry.replace('_', ' ').replace('-', ' ')
                        if (industry_clean.lower() in industry_text_level1.lower() or 
                            industry_clean.lower() in industry_text_level2.lower()):
                            industry_count += 1
                            quality += 2
                            industry_found = True
                            break
                    
                    # Also check detailed industries from config with flexible matching
                    for industry in mei_industries:
                        # Convert config format to match CSV format
                        industry_clean = industry.replace('_', ' ').replace('-', ' ')
                        
                        # Special mappings for common variations
                        industry_variations = [
                            industry_clean,
                            industry_clean.replace(' ', ', '),  # "Agriculture Forestry" -> "Agriculture, Forestry"
                            industry_clean.replace(' ', ' and '),  # Handle "and" connector
                            industry_clean.replace('Forestry Fishing', 'Forestry and Fishing'),
                            industry_clean.replace('Food Services', 'Food Services'),
                            industry_clean.replace('Postal Warehousing', 'Postal and Warehousing'),
                            industry_clean.replace('Gas Water Waste', 'Gas, Water and Waste Services'),
                            industry_clean.replace('Insurance Services', 'Insurance Services'),
                            industry_clean.replace('Scientific Technical', 'Scientific and Technical Services'),
                            industry_clean.replace('Care Social Assistance', 'Care and Social Assistance'),
                            industry_clean.replace('Media Telecommunications', 'Media and Telecommunications'),
                            industry_clean.replace('Support Services', 'Support Services'),
                            industry_clean.replace('Administration Safety', 'Administration and Safety'),
                            industry_clean.replace('Recreation Services', 'Recreation Services'),
                            industry_clean.replace('Hiring Real Estate', 'Hiring and Real Estate Services')
                        ]
                        
                        # Check if any variation matches
                        for variation in industry_variations:
                            if (variation.lower() in industry_text_level1.lower() or 
                                variation.lower() in industry_text_level2.lower()):
                                industry_count += 1
                                quality += 1
                                break
                        else:
                            continue  # Only executed if the inner loop didn't break
                        break  # Break outer loop if match found
                    
                    # Check for sex categories from config (MEI Sex and Age files)
                    sex_categories = self.config.get('demographics', {}).get('sex_categories', [])
                    for sex in sex_categories:
                        if (sex.replace('_', ' ').lower() in industry_text_level1.lower() or
                            sex.replace('_', ' ').lower() in industry_text_level2.lower()):
                            industry_count += 1  # Count as category
                            quality += 1
                            break
                    
                    # Check for detailed age groups from config (MEI Sex and Age files) 
                    age_groups_detailed = self.config.get('demographics', {}).get('age_groups_detailed', [])
                    for age in age_groups_detailed:
                        if (age in adjustment_text_level2 or age in adjustment_text_level3 or
                            age in metric_text or age in metric_text_level3):  # Age might appear in different levels
                            adjustment_count += 1  # Count as category
                            quality += 1
                            break
                    
                    # Check for adjustment types in both levels 2 and 3
                    adjustment_terms = ['Actual', 'Seasonally adjusted', 'Trend']
                    if (any(term in adjustment_text_level2 for term in adjustment_terms) or
                        any(term in adjustment_text_level3 for term in adjustment_terms)):
                        adjustment_count += 1
                        quality += 1
                    
                    # Check for MEI metrics in levels 1 and 3
                    metric_terms = ['Filled jobs', 'Earnings', 'cash', 'accrued', 'Employment']
                    if (any(term in metric_text for term in metric_terms) or
                        any(term in metric_text_level3 for term in metric_terms)):
                        metric_count += 1
                        quality += 1
            
            print(f"    Found {industry_count} industries, {adjustment_count} adjustments, {metric_count} metrics")
            
            return {
                "header_rows": 4,  # Using 4 header rows (0,1,2,3)
                "columns": len(df.columns),
                "quality": quality,
                "sample_df": df,
                "is_mei": True,
                "is_industry_mei": True
            }
            
        except Exception as e:
            print(f"    Warning: MEI industry header detection failed: {e}")
            # Final fallback to standard detection
            return {"header_rows": 0, "columns": 0, "quality": 0}
    
    def detect_lci_header_structure(self, filepath, max_rows=10):
        """Special header detection for LCI files with 3-level structure"""
        print(f"  LCI: Detecting 3-level header structure for {filepath.name}")
        
        try:
            # LCI files typically have 3 header rows
            # Row 1: Title, Row 2: Category, Row 3: Subcategory
            df = pd.read_csv(filepath, header=[0, 1, 2], nrows=max_rows)
            
            # Calculate quality based on structure - LCI files are simple and clean
            quality = 0
            
            # Check if we have the basic LCI structure (2 columns usually)
            if len(df.columns) >= 2:
                quality += 1
            
            # Check for date-like first column (LCI files have quarterly data)
            if len(df) > 0:
                first_col_sample = str(df.iloc[0, 0]) if len(df) > 0 else ""
                if any(char in first_col_sample for char in ['Q', '/', '-']) or first_col_sample.isdigit():
                    quality += 2
            
            # Check for LCI-specific terms in column headers
            for col in df.columns:
                col_str = str(col).lower()
                if any(term in col_str for term in ['labour', 'cost', 'wage', 'salary', 'industry', 'sector', 'occupation']):
                    quality += 1
                    break
            
            print(f"    LCI structure detected with quality {quality}")
            
            return {
                "header_rows": 3,
                "columns": len(df.columns),
                "quality": quality,
                "sample_df": df,
                "is_lci": True
            }
            
        except Exception as e:
            print(f"    Warning: LCI header detection failed: {e}")
            return {"header_rows": 0, "columns": 0, "quality": 0}
    
    def detect_ect_header_structure(self, filepath, max_rows=10):
        """Special header detection for ECT files with 3-level structure"""
        print(f"  ECT: Detecting 3-level header structure for {filepath.name}")
        
        try:
            # ECT files typically have 3-4 header rows
            # Try 4 levels first (for percentage change files), then fall back to 3
            df = None
            header_levels = 4
            
            try:
                # Try 4-level headers first
                df = pd.read_csv(filepath, header=[0, 1, 2, 3], nrows=max_rows)
                if len(df.columns) > 5:  # If we got meaningful columns
                    header_levels = 4
                else:
                    raise Exception("Try 3 levels")
            except:
                # Fall back to 3-level headers
                df = pd.read_csv(filepath, header=[0, 1, 2], nrows=max_rows)
                header_levels = 3
            
            # Calculate quality based on industry categories found
            quality = 0
            industry_count = 0
            
            for col in df.columns:
                if isinstance(col, tuple) and len(col) >= 3:
                    # Check for industry categories from config
                    industry_text = str(col[2]) if len(col) > 2 else ""
                    
                    # Check against ECT categories from config
                    ect_categories = self.config.get('industries', {}).get('ect_categories', [])
                    for category in ect_categories:
                        if category.lower().replace('_', ' ') in industry_text.lower():
                            industry_count += 1
                            quality += 1
                            break
                    
                    # Also check for common ECT terms (industry and transaction types)
                    common_terms = ['consumables', 'durables', 'hospitality', 'services', 'apparel', 'motor', 'fuel', 'rts', 'credit', 'debit', 'core', 'industries']
                    if any(term in industry_text.lower() for term in common_terms):
                        quality += 1
            
            print(f"    Found {industry_count} ECT industry categories")
            
            return {
                "header_rows": header_levels,  # Dynamic header levels (3 or 4)
                "columns": len(df.columns),
                "quality": quality,
                "sample_df": df,
                "is_ect": True,
                "header_levels": header_levels
            }
            
        except Exception as e:
            print(f"    Warning: ECT header detection failed: {e}")
            # Fallback to standard detection
            return {"header_rows": 0, "columns": 0, "quality": 0}
    
    def detect_qem_header_structure(self, filepath, max_rows=10):
        """Special header detection for QEM files with 4-level structure"""
        print(f"  QEM: Detecting 4-level header structure for {filepath.name}")
        
        try:
            # QEM files typically have 4 header rows
            # Row 1: Title, Row 2: Industry, Row 3: Sex, Row 4: Earnings type
            df = pd.read_csv(filepath, header=[0, 1, 2, 3], nrows=max_rows)
            
            # Calculate quality based on industry/sex categories found
            quality = 0
            industry_count = 0
            sex_count = 0
            
            for col in df.columns:
                if isinstance(col, tuple) and len(col) >= 4:
                    # Check for industry categories
                    industry_text = str(col[1]) if len(col) > 1 else ""
                    # Check for sex categories  
                    sex_text = str(col[2]) if len(col) > 2 else ""
                    # Check for earnings types
                    earnings_text = str(col[3]) if len(col) > 3 else ""
                    
                    # Check against QEM sectors from config
                    qem_sectors = self.config.get('industries', {}).get('qem_sectors', [])
                    for sector in qem_sectors:
                        if sector.lower().replace('_', ' ') in industry_text.lower():
                            industry_count += 1
                            quality += 1
                            break
                    
                    # Also check for common QEM industry terms
                    common_industries = ['forestry', 'mining', 'manufacturing', 'construction', 'retail', 'hospitality', 'transport', 'education', 'health']
                    if any(term in industry_text.lower() for term in common_industries):
                        quality += 1
                    
                    # Check for sex categories
                    if any(sex in sex_text.lower() for sex in ['male', 'female', 'both']):
                        sex_count += 1
                        quality += 1
                        
                    # Check for earnings types
                    if any(term in earnings_text.lower() for term in ['hourly', 'ordinary', 'overtime', 'total']):
                        quality += 1
            
            print(f"    Found {industry_count} QEM industries and {sex_count} sex categories")
            
            return {
                "header_rows": 4,  # Using 4 header rows (0,1,2,3)
                "columns": len(df.columns),
                "quality": quality,
                "sample_df": df,
                "is_qem": True
            }
            
        except Exception as e:
            print(f"    Warning: QEM header detection failed: {e}")
            # Fallback to standard detection
            return {"header_rows": 0, "columns": 0, "quality": 0}
    
    def detect_qem_sector_header_structure(self, filepath, max_rows=10):
        """Special header detection for QEM sector files with 5-level structure"""
        print(f"  QEM-SECTOR: Detecting 5-level sector header structure for {filepath.name}")
        
        try:
            # QEM sector files have 5 header rows for percentage change data
            # Row 1: Title, Row 2: Sector, Row 3: Sex, Row 4: Earnings type, Row 5: Change type
            df = pd.read_csv(filepath, header=[0, 1, 2, 3, 4], nrows=max_rows)
            
            # Calculate quality based on sector/sex categories and percentage change terms found
            quality = 0
            sector_count = 0
            sex_count = 0
            change_count = 0
            
            for col in df.columns:
                if isinstance(col, tuple) and len(col) >= 5:
                    # Check for sector categories
                    sector_text = str(col[1]) if len(col) > 1 else ""
                    if any(sector in sector_text.lower() for sector in ['private sector', 'total all sectors', 'public sector']):
                        sector_count += 1
                        quality += 2
                    
                    # Check for sex categories  
                    sex_text = str(col[2]) if len(col) > 2 else ""
                    if any(sex in sex_text.lower() for sex in ['male', 'female', 'both']):
                        sex_count += 1
                        quality += 1
                        
                    # Check for percentage change types
                    change_text = str(col[4]) if len(col) > 4 else ""
                    if any(term in change_text.lower() for term in ['percentage change', 'change from']):
                        change_count += 1
                        quality += 2
            
            print(f"    Found {sector_count} QEM sectors, {sex_count} sex categories, {change_count} change types")
            
            return {
                "header_rows": 5,  # Using 5 header rows (0,1,2,3,4)
                "columns": len(df.columns),
                "quality": quality,
                "sample_df": df,
                "is_qem_sector": True
            }
            
        except Exception as e:
            print(f"    Warning: QEM sector header detection failed: {e}")
            # Fallback to regular QEM detection
            return {"header_rows": 0, "columns": 0, "quality": 0}
    
    def detect_hlf_header_structure(self, filepath, max_rows=10):
        """Special header detection for HLF files with 4-level demographic structure"""
        print(f"  HLF: Detecting 4-level demographic header structure for {filepath.name}")
        
        try:
            # HLF files typically have 4 header rows
            # Row 1: Title, Row 2: Age Groups, Row 3: Regions, Row 4: Labour force metrics
            df = pd.read_csv(filepath, header=[0, 1, 2, 3], nrows=max_rows)
            
            # Calculate quality based on age groups and labour force metrics found
            quality = 0
            age_count = 0
            region_count = 0
            metric_count = 0
            
            for col in df.columns:
                if isinstance(col, tuple) and len(col) >= 4:
                    # Check for age categories
                    age_text = str(col[1]) if len(col) > 1 else ""
                    # Check for region categories  
                    region_text = str(col[2]) if len(col) > 2 else ""
                    # Check for labour force metrics
                    metric_text = str(col[3]) if len(col) > 3 else ""
                    
                    # Check for age groups
                    age_patterns = ['15-24', '25-54', '55', 'Total', 'All Ages']
                    if any(pattern in age_text for pattern in age_patterns):
                        age_count += 1
                        quality += 1
                    
                    # Check for regions
                    region_patterns = ['Auckland', 'Wellington', 'Canterbury', 'Northland', 'Waikato']
                    if any(pattern in region_text for pattern in region_patterns):
                        region_count += 1
                        quality += 1
                    
                    # Check for labour force metrics
                    metric_patterns = ['Employed', 'Unemployed', 'Labour Force', 'Participation', 'Employment Rate', 'Working Age']
                    if any(pattern in metric_text for pattern in metric_patterns):
                        metric_count += 1
                        quality += 1
            
            print(f"    Found {age_count} age groups, {region_count} regions, {metric_count} labour metrics")
            
            return {
                "header_rows": 4,  # Using 4 header rows (0,1,2,3)
                "columns": len(df.columns),
                "quality": quality,
                "sample_df": df,
                "is_hlf": True
            }
            
        except Exception as e:
            print(f"    Warning: HLF header detection failed: {e}")
            # Fallback to standard detection
            return {"header_rows": 0, "columns": 0, "quality": 0}
    
    def standardize_columns(self, df):
        """Clean and standardize column names with demographic preservation"""
        new_columns = []
        
        # Check if this is QEM sector data FIRST (5-level headers with sector/sex/earnings/percentage change)
        is_qem_sector_data = any(isinstance(col, tuple) and len(col) >= 4 and
                                any(sector in str(col).lower() for sector in ['private sector', 'total all sectors', 'public sector']) and
                                any(term in str(col).lower() for term in ['hourly', 'earnings', 'ordinary']) and
                                any(change in str(col).lower() for change in ['percentage change', 'change from'])
                                for col in df.columns)
        
        # Check if this is MEI data (complex 5-level headers for age/region or 4-level for industry) 
        # Only if not QEM sector data
        is_mei_data = not is_qem_sector_data and any(isinstance(col, tuple) and len(col) >= 5 for col in df.columns)
        
        # Check if this is industry MEI data (4-level headers with industry/adjustment/metrics)
        mei_high_level = self.config.get('industries', {}).get('mei_high_level', [])
        is_industry_mei_data = any(isinstance(col, tuple) and len(col) >= 4 and
                                  any(industry.replace('_', ' ').lower() in str(col).lower() for industry in mei_high_level) and
                                  any(adjustment in str(col).lower() for adjustment in ['actual', 'seasonally adjusted', 'trend']) and  
                                  any(metric in str(col).lower() for metric in ['filled jobs', 'earnings', 'cash', 'accrued'])
                                  for col in df.columns)
        
        # Check if this is ECT data (3-level headers with transaction/industry categories)  
        is_ect_data = any(isinstance(col, tuple) and len(col) >= 3 and
                         any(term in str(col).lower() for term in ['consumables', 'durables', 'hospitality', 'services', 'apparel', 'rts', 'credit', 'debit', 'total', 'core', 'industries'])
                         for col in df.columns)
        
        # Check if this is QEM data (4-level headers with industry/sex/earnings) - check before ECT
        is_qem_data = any(isinstance(col, tuple) and len(col) >= 4 and
                         any(term in str(col).lower() for term in ['hourly', 'earnings', 'ordinary', 'overtime', 'employee', 'employment', 'filled', 'jobs']) and
                         any(sex in str(col).lower() for sex in ['male', 'female', 'both'])
                         for col in df.columns)
        
        
        # Check if this is HLF data (4-level headers with age/region/labour force metrics)
        is_hlf_data = any(isinstance(col, tuple) and len(col) >= 4 and
                         any(age_term in str(col).lower() for age_term in ['15-24', '25-54', '55', 'total', 'all ages']) and
                         any(labour_term in str(col).lower() for labour_term in ['employed', 'unemployed', 'labour force', 'participation', 'employment rate', 'working age'])
                         for col in df.columns)
        
        # Check if this is LCI data (3-level headers with labour cost/wage information)
        is_lci_data = any(isinstance(col, tuple) and len(col) >= 3 and
                         any(term in str(col).lower() for term in ['labour', 'cost', 'wage', 'salary', 'industry', 'sector', 'occupation'])
                         for col in df.columns)
        
        # Check if this is demographic data (multi-level headers with demographic groups)
        is_ethnic_data = any(isinstance(col, tuple) and len(col) >= 2 and 
                           any(ethnic in str(col) for ethnic in self.ethnic_groups) 
                           for col in df.columns)
        
        is_sex_data = any(isinstance(col, tuple) and len(col) >= 2 and 
                         any(sex in str(col) for sex in self.sex_categories) 
                         for col in df.columns)
        
        is_age_data = any(isinstance(col, tuple) and len(col) >= 2 and 
                         any(age in str(col) for age in self.age_groups + self.age_groups_detailed) 
                         for col in df.columns)
        
        if is_mei_data:
            # Special handling for MEI employment data structure
            new_columns = self.process_mei_columns(df.columns)
        elif is_industry_mei_data:
            # Special handling for industry MEI data structure - check what type
            # Look for sex/age patterns in columns to decide processing method
            has_sex_age = any(isinstance(col, tuple) and len(col) >= 3 and
                             any(sex.replace('_', ' ').lower() in str(col).lower() 
                                 for sex in self.config.get('demographics', {}).get('sex_categories', [])) and
                             any(age in str(col) 
                                 for age in self.config.get('demographics', {}).get('age_groups_detailed', []))
                             for col in df.columns)
            
            if has_sex_age:
                new_columns = self.process_mei_sex_age_columns(df.columns)  
            else:
                new_columns = self.process_industry_mei_columns(df.columns)
        elif is_qem_sector_data:
            # Special handling for QEM sector earnings data structure (must come before regular QEM)
            new_columns = self.process_qem_sector_columns(df.columns)
        elif is_qem_data:
            # Special handling for QEM earnings data structure
            new_columns = self.process_qem_columns(df.columns)
        elif is_hlf_data:
            # Special handling for HLF labour force data structure
            new_columns = self.process_hlf_columns(df.columns)
        elif is_ect_data:
            # Special handling for ECT transaction data structure
            new_columns = self.process_ect_columns(df.columns)
        elif is_ethnic_data:
            # Special handling for ethnic unemployment data structure
            new_columns = self.process_ethnic_columns(df.columns)
        elif is_sex_data or is_age_data:
            # Special handling for sex/age demographic data
            new_columns = self.process_demographic_columns(df.columns, is_sex_data, is_age_data)
        elif is_lci_data:
            # Special handling for LCI labour cost index data
            new_columns = self.process_lci_columns(df.columns)
        else:
            # Standard column processing (for ECT, BUO, and other multi-level headers)
            for i, col in enumerate(df.columns):
                if isinstance(col, tuple):  # Multi-level header
                    # Extract meaningful parts, filtering out empty/unnamed elements
                    meaningful_parts = []
                    for part in col:
                        part_str = str(part).strip()
                        if (part_str and 
                            part_str not in ['', 'nan', 'NaN', '..'] and
                            'Unnamed' not in part_str and
                            'level_' not in part_str):
                            meaningful_parts.append(part_str)
                    
                    if meaningful_parts:
                        col_str = '_'.join(meaningful_parts)
                    else:
                        col_str = f'column_{i}'
                else:
                    col_str = str(col).strip()
                
                # Clean column name
                col_str = re.sub(r'\s+', '_', col_str)  # Spaces to underscores
                col_str = re.sub(r'[^\w\s-]', '_', col_str)  # Special chars to underscores
                col_str = col_str.replace('-', '_').replace('/', '_').replace('(', '_').replace(')', '_')
                col_str = re.sub(r'_+', '_', col_str).strip('_')  # Remove multiple underscores
                
                # Handle empty or problematic names
                if not col_str or col_str in ['nan', 'NaN', '..', 'Unnamed']:
                    col_str = f'column_{i}'
                    
                new_columns.append(col_str)
        
        df.columns = new_columns
        return df
    
    def process_ethnic_columns(self, columns):
        """Process multi-level ethnic columns to preserve demographic unemployment rates"""
        print("  ETHNIC: Processing ethnic unemployment data structure")
        
        new_columns = []
        current_ethnic = None
        current_region = None
        
        for i, col in enumerate(columns):
            if isinstance(col, tuple) and len(col) >= 4:
                level1 = str(col[1]).strip() if len(col) > 1 else ""  # Ethnic group
                level2 = str(col[2]).strip() if len(col) > 2 else ""  # Region  
                level3 = str(col[3]).strip() if len(col) > 3 else ""  # Metric
                
                # Update current ethnic group if we find one
                for ethnic in self.ethnic_groups:
                    if ethnic in level1:
                        current_ethnic = ethnic
                        break
                
                # Update current region if we find one (not "Unnamed")
                if level2 and 'Unnamed' not in level2 and level2.strip():
                    current_region = level2
                
                # Build column name
                if current_ethnic and level3:
                    # Use current region or "Total" as fallback
                    region = current_region if current_region and 'Unnamed' not in current_region else "Total"
                    metric = level3
                    
                    # Special handling for unemployment rate
                    if "Unemployment Rate" in metric:
                        col_name = f"{current_ethnic}_{region}_Unemployment_Rate"
                    elif "Employment Rate" in metric:
                        col_name = f"{current_ethnic}_{region}_Employment_Rate"
                    elif "Labour Force Participation Rate" in metric:
                        col_name = f"{current_ethnic}_{region}_Labour_Force_Participation_Rate"
                    else:
                        # Clean metric name
                        clean_metric = metric.replace(' ', '_')
                        col_name = f"{current_ethnic}_{region}_{clean_metric}"
                        
                    # Clean the name
                    col_name = re.sub(r'\s+', '_', col_name)
                    col_name = re.sub(r'[^\w\s-]', '_', col_name)
                    col_name = col_name.replace('-', '_').replace('/', '_')
                    col_name = re.sub(r'_+', '_', col_name).strip('_')
                    
                else:
                    # Fallback to standard processing for non-ethnic columns
                    col_parts = [str(part).strip() for part in col if str(part).strip() and 'Unnamed' not in str(part) and part.strip()]
                    col_name = '_'.join(col_parts) if col_parts else f'column_{i}'
                    col_name = re.sub(r'\s+', '_', col_name)
                    col_name = re.sub(r'[^\w\s-]', '_', col_name)
                    col_name = col_name.replace('-', '_').replace('/', '_')
                    col_name = re.sub(r'_+', '_', col_name).strip('_')
                    
            else:
                # Handle non-tuple columns (like date column)
                if isinstance(col, tuple):
                    col_parts = [str(part).strip() for part in col if str(part).strip() and 'Unnamed' not in str(part)]
                    col_name = '_'.join(col_parts) if col_parts else f'column_{i}'
                else:
                    col_name = str(col).strip()
                
                col_name = re.sub(r'\s+', '_', col_name)
                col_name = re.sub(r'[^\w\s-]', '_', col_name)
                col_name = col_name.replace('-', '_').replace('/', '_')
                col_name = re.sub(r'_+', '_', col_name).strip('_')
                
                if not col_name or col_name in ['nan', 'NaN', '..']:
                    col_name = f'column_{i}'
            
            new_columns.append(col_name)
        
        # Log the unemployment rate columns found
        unemployment_cols = [col for col in new_columns if 'Unemployment_Rate' in col]
        if unemployment_cols:
            print(f"    Found {len(unemployment_cols)} ethnic unemployment rate columns")
            print(f"    Sample: {unemployment_cols[:5]}")
        
        return new_columns
    
    def process_demographic_columns(self, columns, is_sex_data, is_age_data):
        """Process multi-level sex/age demographic columns to create unemployment rates"""
        print("  DEMOGRAPHIC: Processing sex/age unemployment data structure")
        
        new_columns = []
        current_demographic = None
        current_region = None
        
        for i, col in enumerate(columns):
            if isinstance(col, tuple) and len(col) >= 2:
                # Extract parts from multi-level header
                level1 = str(col[0]).strip() if len(col) > 0 else ""
                level2 = str(col[1]).strip() if len(col) > 1 else ""  
                level3 = str(col[2]).strip() if len(col) > 2 else ""
                level4 = str(col[3]).strip() if len(col) > 3 else ""
                
                # Detect sex demographics
                if is_sex_data:
                    for sex in self.sex_categories:
                        if sex in level1 or sex in level2:
                            current_demographic = sex
                            break
                
                # Detect age demographics  
                if is_age_data:
                    all_age_groups = self.age_groups + self.age_groups_detailed
                    for age in all_age_groups:
                        if age in level1 or age in level2 or age in level3:
                            current_demographic = age.replace(' ', '_').replace('+', '_Plus')
                            break
                
                # Detect region
                region_found = False
                for level in [level1, level2, level3, level4]:
                    if level and 'Unnamed' not in level and level.strip():
                        # Check if it contains region names
                        for region_list in self.config.get('regions', {}).values():
                            if isinstance(region_list, list):
                                for region in region_list:
                                    if region.replace('_', ' ') in level or region.replace(' ', '_') in level:
                                        current_region = region.replace(' ', '_')
                                        region_found = True
                                        break
                                if region_found:
                                    break
                        if region_found:
                            break
                
                # Build column name for unemployment metrics
                if current_demographic:
                    region = current_region if current_region else "Total"
                    
                    # Look for unemployment-related metrics
                    metric_found = False
                    for level in [level1, level2, level3, level4]:
                        if 'Unemployment' in level and 'Rate' in level:
                            col_name = f"{current_demographic}_{region}_Unemployment_Rate"
                            metric_found = True
                            break
                        elif 'Employment' in level and 'Rate' in level:
                            col_name = f"{current_demographic}_{region}_Employment_Rate"
                            metric_found = True
                            break
                        elif 'Labour' in level and 'Force' in level and 'Participation' in level:
                            col_name = f"{current_demographic}_{region}_Labour_Force_Participation_Rate"
                            metric_found = True
                            break
                    
                    if not metric_found:
                        # Use the last meaningful level as metric
                        meaningful_level = None
                        for level in reversed([level1, level2, level3, level4]):
                            if level and 'Unnamed' not in level and level.strip():
                                meaningful_level = level.replace(' ', '_')
                                break
                        col_name = f"{current_demographic}_{region}_{meaningful_level}" if meaningful_level else f"column_{i}"
                    
                    # Clean the name
                    col_name = re.sub(r'\s+', '_', col_name)
                    col_name = re.sub(r'[^\w\s-]', '_', col_name)
                    col_name = col_name.replace('-', '_').replace('/', '_')
                    col_name = re.sub(r'_+', '_', col_name).strip('_')
                    
                else:
                    # Fallback to standard processing
                    col_parts = [str(part).strip() for part in col if str(part).strip() and 'Unnamed' not in str(part)]
                    col_name = '_'.join(col_parts) if col_parts else f'column_{i}'
                    col_name = re.sub(r'\s+', '_', col_name)
                    col_name = re.sub(r'[^\w\s-]', '_', col_name)
                    col_name = col_name.replace('-', '_').replace('/', '_')
                    col_name = re.sub(r'_+', '_', col_name).strip('_')
            else:
                # Handle single-level columns
                col_name = str(col).strip()
                col_name = re.sub(r'\s+', '_', col_name)
                col_name = re.sub(r'[^\w\s-]', '_', col_name)
                col_name = col_name.replace('-', '_').replace('/', '_')
                col_name = re.sub(r'_+', '_', col_name).strip('_')
                
                if not col_name or col_name in ['nan', 'NaN', '..']:
                    col_name = f'column_{i}'
            
            new_columns.append(col_name)
        
        # Log the unemployment rate columns found
        unemployment_cols = [col for col in new_columns if 'Unemployment_Rate' in col]
        if unemployment_cols:
            demo_type = "sex/age" if is_sex_data or is_age_data else "demographic"
            print(f"    Found {len(unemployment_cols)} {demo_type} unemployment rate columns")
            print(f"    Sample: {unemployment_cols[:5]}")
        
        return new_columns
    
    def process_lci_columns(self, columns):
        """Process LCI 3-level headers: Title/Category/Subcategory"""
        print("  LCI: Processing labour cost index data structure")
        
        new_columns = []
        
        for i, col in enumerate(columns):
            if i == 0:
                # First column is usually the date/period column
                new_columns.append('date')
            elif isinstance(col, tuple) and len(col) >= 3:
                # Extract meaningful parts from the tuple
                level0 = str(col[0]).strip() if len(col) > 0 else ""
                level1 = str(col[1]).strip() if len(col) > 1 else ""
                level2 = str(col[2]).strip() if len(col) > 2 else ""
                
                # Build column name from non-empty, non-unnamed parts
                col_parts = []
                for level in [level0, level1, level2]:
                    if (level and 'Unnamed' not in level and 
                        level != ' ' and level.strip()):
                        # Clean and add the part
                        clean_part = re.sub(r'[^\w\s-]', ' ', level)
                        clean_part = re.sub(r'\s+', '_', clean_part).strip('_')
                        if clean_part:
                            col_parts.append(clean_part)
                
                if col_parts:
                    col_name = '_'.join(col_parts)
                    # Additional cleaning
                    col_name = col_name.replace('__', '_').strip('_')
                    col_name = col_name[:100]  # Limit length
                else:
                    col_name = f'lci_value_{i}'
                    
                new_columns.append(col_name)
            else:
                # Handle simple column names
                col_str = str(col).strip()
                if col_str and 'Unnamed' not in col_str:
                    clean_name = re.sub(r'[^\w\s-]', ' ', col_str)
                    clean_name = re.sub(r'\s+', '_', clean_name).strip('_')
                    new_columns.append(clean_name if clean_name else f'lci_value_{i}')
                else:
                    new_columns.append(f'lci_value_{i}')
        
        print(f"    Generated {len(new_columns)} LCI column names")
        print(f"    Sample: {new_columns[:3]}")
        
        return new_columns
    
    def process_mei_columns(self, columns):
        """Process MEI complex 5-level headers: Title/Category/Age/Region/Measurement"""
        print("  MEI: Processing complex employment data structure")
        
        new_columns = []
        current_age_group = None
        
        for i, col in enumerate(columns):
            if isinstance(col, tuple) and len(col) >= 3:
                # MEI structure: (Title, Category, Age, Region, Measurement)
                # Age group is typically in col[2], Region in col[3]
                parts = [str(part).strip() for part in col if str(part).strip()]
                
                age_part = str(col[2]).strip() if len(col) > 2 else ""
                region_part = str(col[3]).strip() if len(col) > 3 else ""
                
                # Check if this column defines a new age group
                found_age = None
                if age_part and age_part not in ['', 'nan', 'NaN', 'Unnamed']:
                    # Look for age patterns in the age part
                    for age in self.age_groups_detailed:
                        if age in age_part or age_part == age:
                            found_age = age
                            current_age_group = age
                            break
                    
                    # Also check for direct age patterns like "15-19", "65"
                    if not found_age:
                        if re.match(r'\d+-\d+', age_part):
                            found_age = age_part
                            current_age_group = age_part
                        elif re.match(r'^\d+$', age_part):  # Just a number like "65"
                            found_age = age_part
                            current_age_group = age_part
                
                # Find region
                region = None
                if region_part and region_part not in ['', 'nan', 'NaN', 'Unnamed']:
                    # Direct region match
                    region = region_part
                    # Clean up region name
                    if "'" in region:
                        region = region.replace("'", "_")
                
                # Build column name
                col_parts = []
                
                # Use current age group (carries forward)
                if current_age_group:
                    clean_age = current_age_group.replace('-', '_').replace('+', '_Plus').replace("'", "_")
                    col_parts.append(clean_age)
                
                # Add region
                if region:
                    clean_region = region.replace(' ', '_').replace('-', '_').replace("'", "_")
                    col_parts.append(clean_region)
                
                # Add employment context
                col_parts.append('Employment')
                
                # Construct final name
                if col_parts:
                    col_name = '_'.join(col_parts)
                else:
                    # Fallback - use meaningful non-empty parts
                    meaningful_parts = [p for p in parts if p and 'Unnamed' not in p and p.strip()]
                    if meaningful_parts:
                        col_name = '_'.join([p.replace(' ', '_').replace('-', '_') for p in meaningful_parts[:3]])
                    else:
                        col_name = f'MEI_column_{i}'
                
                # Clean the final name
                col_name = re.sub(r'\s+', '_', col_name)
                col_name = re.sub(r'[^\w\s-]', '_', col_name)
                col_name = col_name.replace('-', '_').replace('/', '_').replace("'", "_")
                col_name = re.sub(r'_+', '_', col_name).strip('_')
                
            else:
                # Handle non-tuple or simple columns (like date column)
                if isinstance(col, tuple):
                    # Extract non-empty, non-unnamed parts
                    meaningful_parts = []
                    for part in col:
                        part_str = str(part).strip()
                        if (part_str and 
                            'Unnamed' not in part_str and 
                            part_str not in ['', 'nan', 'NaN'] and
                            'level_' not in part_str):
                            meaningful_parts.append(part_str)
                    
                    if meaningful_parts:
                        col_name = '_'.join([p.replace(' ', '_').replace('-', '_') for p in meaningful_parts])
                    else:
                        col_name = f'column_{i}'
                else:
                    col_name = str(col).strip()
                
                # Clean the name
                if col_name and col_name not in ['', 'nan', 'NaN']:
                    col_name = re.sub(r'\s+', '_', col_name)
                    col_name = re.sub(r'[^\w\s-]', '_', col_name)
                    col_name = col_name.replace('-', '_').replace('/', '_').replace("'", "_")
                    col_name = re.sub(r'_+', '_', col_name).strip('_')
                else:
                    col_name = f'column_{i}'
            
            new_columns.append(col_name)
        
        # Log the employment columns found
        employment_cols = [col for col in new_columns if 'employment' in col.lower()]
        if employment_cols:
            print(f"    Found {len(employment_cols)} MEI employment columns")
            print(f"    Sample: {employment_cols[:5]}")
        
        return new_columns
    
    def process_industry_mei_columns(self, columns):
        """Process industry MEI 4-level headers: Title/Industry/Adjustment/Metric"""
        print("  MEI-INDUSTRY: Processing industry-based employment data structure")
        
        new_columns = []
        current_industry = None
        current_adjustment = None
        
        for i, col in enumerate(columns):
            if isinstance(col, tuple) and len(col) >= 4:
                # MEI-Industry structure: (Title, Industry, Adjustment, Metric)
                title = str(col[0]).strip() if len(col) > 0 else ""
                industry = str(col[1]).strip() if len(col) > 1 else ""
                adjustment = str(col[2]).strip() if len(col) > 2 else ""
                metric = str(col[3]).strip() if len(col) > 3 else ""
                
                # Track current industry (only appears once in MultiIndex header)
                if industry and industry not in ['', 'nan', 'NaN', 'Unnamed'] and 'level' not in industry:
                    current_industry = industry
                
                # Track current adjustment type (only appears once per industry)
                if adjustment and adjustment not in ['', 'nan', 'NaN', 'Unnamed'] and 'level' not in adjustment:
                    current_adjustment = adjustment
                
                # Build column name using tracked values
                col_parts = []
                
                # Add industry
                if current_industry:
                    clean_industry = current_industry.replace(' ', '_').replace('-', '_')
                    # Use standardized names from config
                    mei_high_level = self.config.get('industries', {}).get('mei_high_level', [])
                    for config_industry in mei_high_level:
                        config_clean = config_industry.replace('_', ' ').replace('-', ' ')
                        if config_clean.lower() in current_industry.lower():
                            clean_industry = config_industry
                            break
                    col_parts.append(clean_industry)
                
                # Add adjustment type
                if current_adjustment:
                    clean_adjustment = current_adjustment.replace(' ', '_')
                    col_parts.append(clean_adjustment)
                
                # Add metric
                if metric and metric not in ['', 'nan', 'NaN', 'Unnamed']:
                    clean_metric = metric.replace(' ', '_').replace('-', '_')
                    # Standardize metric names
                    if 'Filled jobs' in metric:
                        clean_metric = 'Filled_Jobs'
                    elif 'Earnings - cash' in metric:
                        clean_metric = 'Earnings_Cash'
                    elif 'Earnings - accrued' in metric:
                        clean_metric = 'Earnings_Accrued'
                    col_parts.append(clean_metric)
                
                # Construct final name
                if col_parts:
                    col_name = '_'.join(col_parts)
                else:
                    # Fallback - use meaningful parts
                    meaningful_parts = []
                    for part in col:
                        part_str = str(part).strip()
                        if (part_str and 
                            'Unnamed' not in part_str and 
                            part_str not in ['', 'nan', 'NaN']):
                            meaningful_parts.append(part_str)
                    
                    if meaningful_parts:
                        col_name = '_'.join([p.replace(' ', '_').replace('-', '_') for p in meaningful_parts])
                    else:
                        col_name = f'MEI_Industry_column_{i}'
                
                # Clean the final name
                col_name = re.sub(r'\s+', '_', col_name)
                col_name = re.sub(r'[^\w\s-]', '_', col_name)
                col_name = col_name.replace('-', '_').replace('/', '_')
                col_name = re.sub(r'_+', '_', col_name).strip('_')
                
            else:
                # Handle non-tuple or simple columns (like date column)
                if isinstance(col, tuple):
                    meaningful_parts = []
                    for part in col:
                        part_str = str(part).strip()
                        if (part_str and 
                            'Unnamed' not in part_str and 
                            part_str not in ['', 'nan', 'NaN'] and
                            'level_' not in part_str):
                            meaningful_parts.append(part_str)
                    
                    if meaningful_parts:
                        col_name = '_'.join([p.replace(' ', '_').replace('-', '_') for p in meaningful_parts])
                    else:
                        col_name = f'column_{i}'
                else:
                    col_name = str(col).strip()
                
                col_name = re.sub(r'\s+', '_', col_name)
                col_name = re.sub(r'[^\w\s-]', '_', col_name)
                col_name = col_name.replace('-', '_').replace('/', '_')
                col_name = re.sub(r'_+', '_', col_name).strip('_')
                
                if not col_name or col_name in ['nan', 'NaN', '..']:
                    col_name = f'column_{i}'
            
            new_columns.append(col_name)
        
        # Log the industry employment columns found
        industry_cols = [col for col in new_columns if any(term in col for term in ['Filled_Jobs', 'Earnings', 'Industries'])]
        if industry_cols:
            print(f"    Found {len(industry_cols)} MEI industry columns")
            print(f"    Sample: {industry_cols[:5]}")
        
        return new_columns
    
    def process_qem_sector_columns(self, columns):
        """Process QEM sector 4-level headers: Title/Sector/Sex/Earnings_Type + Percentage_Change"""
        print("  QEM-SECTOR: Processing sector earnings percentage change data structure")
        
        new_columns = []
        current_sector = None
        current_sex = None
        current_earnings_type = None
        
        for i, col in enumerate(columns):
            if isinstance(col, tuple) and len(col) >= 4:
                # QEM Sector structure: (Title, Sector, Sex, Earnings_Type, Change_Type)
                title = str(col[0]).strip() if len(col) > 0 else ""
                sector = str(col[1]).strip() if len(col) > 1 else ""
                sex = str(col[2]).strip() if len(col) > 2 else ""  
                earnings_type = str(col[3]).strip() if len(col) > 3 else ""
                change_type = str(col[4]).strip() if len(col) > 4 else ""
                
                # Track current context values (only update when meaningful)
                if sector and sector not in ['', 'nan', 'NaN', 'Unnamed'] and 'level' not in sector:
                    current_sector = sector
                if sex and sex not in ['', 'nan', 'NaN', 'Unnamed'] and 'level' not in sex:
                    current_sex = sex
                if earnings_type and earnings_type not in ['', 'nan', 'NaN', 'Unnamed'] and 'level' not in earnings_type:
                    current_earnings_type = earnings_type
                
                # Build column name using tracked values
                col_parts = []
                
                # Add sector
                if current_sector:
                    clean_sector = current_sector.replace(' ', '_')
                    if 'Private Sector' in current_sector:
                        clean_sector = 'Private_Sector'
                    elif 'Total All Sectors' in current_sector:
                        clean_sector = 'Total_All_Sectors'
                    elif 'Public Sector' in current_sector:
                        clean_sector = 'Public_Sector'
                    col_parts.append(clean_sector)
                
                # Add sex using tracked value
                if current_sex:
                    if 'Total Both Sexes' in current_sex or 'Both Sexes' in current_sex:
                        col_parts.append('Total_Both_Sexes')
                    elif 'Male' in current_sex and 'Female' not in current_sex:
                        col_parts.append('Male')
                    elif 'Female' in current_sex:
                        col_parts.append('Female')
                
                # Add earnings type using tracked value
                if current_earnings_type:
                    if 'Ordinary Time Hourly' in current_earnings_type:
                        col_parts.append('Ordinary_Time_Hourly')
                    elif 'Overtime Hourly' in current_earnings_type:
                        col_parts.append('Overtime_Hourly')
                    else:
                        clean_earnings = current_earnings_type.replace(' ', '_')
                        col_parts.append(clean_earnings)
                
                # Add percentage change type from actual column content
                if change_type and change_type not in ['', 'nan', 'NaN', 'Unnamed']:
                    if 'same period previous year' in change_type or 'annual' in change_type.lower():
                        col_parts.append('Annual_Percentage_Change')
                    elif 'previous period' in change_type or 'quarterly' in change_type.lower():
                        col_parts.append('Period_Percentage_Change')
                    else:
                        change_clean = change_type.replace(' ', '_').replace('from', '').strip('_')
                        col_parts.append(f'{change_clean}_Change')
                else:
                    # Fallback: detect from position pattern
                    if i > 0 and (i-1) % 2 == 0:  # Odd columns after first
                        col_parts.append('Annual_Percentage_Change')
                    else:  # Even columns after first
                        col_parts.append('Period_Percentage_Change')
                
                # Construct final name
                if col_parts:
                    col_name = '_'.join(col_parts)
                else:
                    # Fallback - use meaningful parts
                    meaningful_parts = []
                    for part in col:
                        part_str = str(part).strip()
                        if (part_str and 
                            'Unnamed' not in part_str and 
                            part_str not in ['', 'nan', 'NaN']):
                            meaningful_parts.append(part_str)
                    
                    if meaningful_parts:
                        col_name = '_'.join([p.replace(' ', '_') for p in meaningful_parts])
                    else:
                        col_name = f'QEM_Sector_column_{i}'
                
                # Clean the final name
                col_name = re.sub(r'\s+', '_', col_name)
                col_name = re.sub(r'[^\w\s-]', '_', col_name)
                col_name = col_name.replace('-', '_').replace('/', '_')
                col_name = re.sub(r'_+', '_', col_name).strip('_')
                
            else:
                # Handle non-tuple or simple columns (like date column)
                if isinstance(col, tuple):
                    meaningful_parts = []
                    for part in col:
                        part_str = str(part).strip()
                        if (part_str and 
                            'Unnamed' not in part_str and 
                            part_str not in ['', 'nan', 'NaN'] and
                            'level_' not in part_str):
                            meaningful_parts.append(part_str)
                    
                    if meaningful_parts:
                        col_name = '_'.join([p.replace(' ', '_') for p in meaningful_parts])
                    else:
                        col_name = f'column_{i}'
                else:
                    col_name = str(col).strip()
                
                col_name = re.sub(r'\s+', '_', col_name)
                col_name = re.sub(r'[^\w\s-]', '_', col_name)
                col_name = col_name.replace('-', '_').replace('/', '_')
                col_name = re.sub(r'_+', '_', col_name).strip('_')
                
                if not col_name or col_name in ['nan', 'NaN', '..']:
                    col_name = f'column_{i}'
            
            new_columns.append(col_name)
        
        # Log the QEM sector columns found
        sector_cols = [col for col in new_columns if any(term in col for term in ['Private_Sector', 'Total_All_Sectors', 'Percentage_Change'])]
        if sector_cols:
            print(f"    Found {len(sector_cols)} QEM sector columns")
            print(f"    Sample: {sector_cols[:5]}")
        
        return new_columns
    
    def process_mei_sex_age_columns(self, columns):
        """Process MEI Sex/Age 4+ level headers: Title/Metric/Sex/Age/Adjustment"""
        print("  MEI-SEX-AGE: Processing sex and age employment data structure")
        
        new_columns = []
        current_sex = None
        current_age = None
        
        for i, col in enumerate(columns):
            if isinstance(col, tuple) and len(col) >= 4:
                # MEI Sex/Age structure: (Title, Metric, Sex, Age, Adjustment)
                title = str(col[0]).strip() if len(col) > 0 else ""
                metric = str(col[1]).strip() if len(col) > 1 else ""
                sex = str(col[2]).strip() if len(col) > 2 else ""
                age = str(col[3]).strip() if len(col) > 3 else ""
                adjustment = str(col[4]).strip() if len(col) > 4 else ""
                
                # Track current sex (only appears once in MultiIndex header)
                sex_categories = self.config.get('demographics', {}).get('sex_categories', [])
                for sex_cat in sex_categories:
                    if sex_cat.replace('_', ' ').lower() in sex.lower() and 'unnamed' not in sex.lower():
                        current_sex = sex_cat
                        break
                
                # Track current age (only appears once per sex group) 
                age_groups = self.config.get('demographics', {}).get('age_groups_detailed', [])
                for age_group in age_groups:
                    if age_group in age and 'unnamed' not in age.lower():
                        current_age = age_group.replace('-', '_to_').replace('+', '_Plus')
                        break
                
                # Build column name using tracked values
                col_parts = []
                
                # Add sex
                if current_sex:
                    col_parts.append(current_sex.replace('_', '_'))
                
                # Add age group
                if current_age:
                    col_parts.append(f"Age_{current_age}")
                
                # Add metric (usually "Filled jobs")
                if metric and metric not in ['', 'nan', 'NaN', 'Unnamed']:
                    clean_metric = metric.replace(' ', '_').replace('-', '_')
                    if 'Filled jobs' in metric:
                        clean_metric = 'Filled_Jobs'
                    elif 'Employment' in metric:
                        clean_metric = 'Employment'
                    col_parts.append(clean_metric)
                
                # Add adjustment type (usually "Actual")
                if adjustment and adjustment not in ['', 'nan', 'NaN', 'Unnamed']:
                    col_parts.append(adjustment.replace(' ', '_'))
                
                # Construct final name
                if col_parts:
                    col_name = '_'.join(col_parts)
                else:
                    # Fallback - use meaningful parts
                    meaningful_parts = []
                    for part in col:
                        part_str = str(part).strip()
                        if (part_str and 
                            'Unnamed' not in part_str and 
                            part_str not in ['', 'nan', 'NaN']):
                            meaningful_parts.append(part_str)
                    
                    if meaningful_parts:
                        col_name = '_'.join([p.replace(' ', '_').replace('-', '_') for p in meaningful_parts])
                    else:
                        col_name = f'MEI_SexAge_column_{i}'
                
                # Clean the final name
                col_name = re.sub(r'\s+', '_', col_name)
                col_name = re.sub(r'[^\w\s-]', '_', col_name)
                col_name = col_name.replace('-', '_').replace('/', '_')
                col_name = re.sub(r'_+', '_', col_name).strip('_')
                
            else:
                # Handle non-tuple or simple columns (like date column)
                if isinstance(col, tuple):
                    meaningful_parts = []
                    for part in col:
                        part_str = str(part).strip()
                        if (part_str and 
                            'Unnamed' not in part_str and 
                            part_str not in ['', 'nan', 'NaN'] and
                            'level_' not in part_str):
                            meaningful_parts.append(part_str)
                    
                    if meaningful_parts:
                        col_name = '_'.join([p.replace(' ', '_').replace('-', '_') for p in meaningful_parts])
                    else:
                        col_name = f'column_{i}'
                else:
                    col_name = str(col).strip()
                
                col_name = re.sub(r'\s+', '_', col_name)
                col_name = re.sub(r'[^\w\s-]', '_', col_name)
                col_name = col_name.replace('-', '_').replace('/', '_')
                col_name = re.sub(r'_+', '_', col_name).strip('_')
                
                if not col_name or col_name in ['nan', 'NaN', '..']:
                    col_name = f'column_{i}'
            
            new_columns.append(col_name)
        
        # Log the sex/age employment columns found
        sexage_cols = [col for col in new_columns if any(term in col for term in ['Male', 'Female', 'Age', 'Filled_Jobs'])]
        if sexage_cols:
            print(f"    Found {len(sexage_cols)} MEI sex/age columns")
            print(f"    Sample: {sexage_cols[:5]}")
        
        return new_columns
    
    def process_ect_columns(self, columns):
        """Process ECT 3/4-level headers: Title/Measurement/Industry or Title/Measurement/SubType/Industry"""
        print("  ECT: Processing electronic card transaction data structure")
        
        new_columns = []
        current_adjustment = None
        current_category = None
        
        for i, col in enumerate(columns):
            if isinstance(col, tuple) and len(col) >= 3:
                # ECT structure: (Title, Measurement_Type, Industry) for 3-level
                # or (Title, Measurement_Type, SubType, Industry) for 4-level  
                title = str(col[0]).strip() if len(col) > 0 else ""
                measurement = str(col[1]).strip() if len(col) > 1 else ""
                subtype = str(col[2]).strip() if len(col) > 2 else ""
                industry = str(col[3]).strip() if len(col) > 3 else str(col[2]).strip() if len(col) > 2 else ""
                
                # Track current adjustment/measurement type (like "Actual", "Seasonally adjusted")
                if measurement and measurement not in ['', 'nan', 'NaN', 'Unnamed'] and 'level' not in measurement:
                    current_adjustment = measurement
                
                # Track current category/industry (like "RTS total industries", "Mean transaction value") 
                if len(col) >= 4:
                    if subtype and subtype not in ['', 'nan', 'NaN', 'Unnamed'] and 'level' not in subtype:
                        current_category = subtype
                else:
                    if industry and industry not in ['', 'nan', 'NaN', 'Unnamed'] and 'level' not in industry:
                        current_category = industry
                
                # Build column name using tracked values
                col_parts = []
                
                # Add current adjustment type
                if current_adjustment:
                    if current_adjustment in ['Actual', 'Seasonally adjusted', 'Trend']:
                        col_parts.append(current_adjustment.replace(' ', '_'))
                
                # Add current category/industry
                if current_category:
                    clean_category = current_category.replace(' ', '_').replace('.', '').replace('/', '_')
                    # Clean up common transaction terms
                    if 'RTS_total_industries' in clean_category:
                        clean_category = 'RTS_Total_Industries'
                    elif 'RTS_core_industries' in clean_category:
                        clean_category = 'RTS_Core_Industries'
                    elif clean_category == 'Total':
                        clean_category = 'Total_Transactions'
                    elif clean_category == 'Credit':
                        clean_category = 'Credit_Card'
                    elif clean_category == 'Debit':
                        clean_category = 'Debit_Card'
                    elif 'Mean_transaction_value' in clean_category:
                        clean_category = 'Mean_Transaction_Value'
                    elif 'Mean_value_of_transaction_per_person' in clean_category:
                        clean_category = 'Mean_Value_Per_Person'
                    elif 'Mean_number_of_transactions_per_person' in clean_category:
                        clean_category = 'Mean_Transactions_Per_Person'
                    col_parts.append(clean_category)
                
                # For 4-level headers, add the final metric type
                if len(col) >= 4 and industry and industry not in ['', 'nan', 'NaN', 'Unnamed']:
                    clean_metric = industry.replace(' ', '_').replace('.', '').replace('/', '_')
                    # Handle percentage change and value types
                    if 'Percentage_change_from_same_period_previous_year' in clean_metric:
                        clean_metric = 'Annual_Percentage_Change'
                    elif 'Percentage_change_from_previous_period' in clean_metric:
                        clean_metric = 'Period_Percentage_Change'
                    elif 'Proportion' in clean_metric and '%' in clean_metric:
                        clean_metric = 'Proportion_Percent'
                    elif 'Value' in clean_metric and '$' in clean_metric:
                        clean_metric = 'Value_Dollars'
                    col_parts.append(clean_metric)
                
                # Construct final name
                if col_parts:
                    col_name = '_'.join(col_parts)
                else:
                    # Fallback - use meaningful parts
                    meaningful_parts = []
                    for part in col:
                        part_str = str(part).strip()
                        if (part_str and 
                            'Unnamed' not in part_str and 
                            part_str not in ['', 'nan', 'NaN']):
                            meaningful_parts.append(part_str)
                    
                    if meaningful_parts:
                        col_name = '_'.join([p.replace(' ', '_').replace('.', '').replace('/', '_') for p in meaningful_parts])
                    else:
                        col_name = f'ECT_column_{i}'
                
                # Clean the final name
                col_name = re.sub(r'\s+', '_', col_name)
                col_name = re.sub(r'[^\w\s-]', '_', col_name)
                col_name = col_name.replace('-', '_').replace('/', '_')
                col_name = re.sub(r'_+', '_', col_name).strip('_')
                
            else:
                # Handle non-tuple or simple columns (like date column)
                if isinstance(col, tuple):
                    meaningful_parts = []
                    for part in col:
                        part_str = str(part).strip()
                        if (part_str and 
                            'Unnamed' not in part_str and 
                            part_str not in ['', 'nan', 'NaN'] and
                            'level_' not in part_str):
                            meaningful_parts.append(part_str)
                    
                    if meaningful_parts:
                        col_name = '_'.join([p.replace(' ', '_').replace('-', '_') for p in meaningful_parts])
                    else:
                        col_name = f'column_{i}'
                else:
                    col_name = str(col).strip()
                
                # Clean the name
                if col_name and col_name not in ['', 'nan', 'NaN']:
                    col_name = re.sub(r'\s+', '_', col_name)
                    col_name = re.sub(r'[^\w\s-]', '_', col_name)
                    col_name = col_name.replace('-', '_').replace('/', '_')
                    col_name = re.sub(r'_+', '_', col_name).strip('_')
                else:
                    col_name = f'column_{i}'
            
            new_columns.append(col_name)
        
        # Log the transaction columns found
        transaction_cols = [col for col in new_columns if any(word in col.lower() for word in ['consumables', 'durables', 'hospitality', 'services', 'apparel', 'motor', 'fuel'])]
        if transaction_cols:
            print(f"    Found {len(transaction_cols)} ECT transaction columns")
            print(f"    Sample: {transaction_cols[:5]}")
        
        return new_columns
    
    def process_qem_columns(self, columns):
        """Process QEM 4-level headers: Title/Industry/Sex/Earnings_Type"""
        print("  QEM: Processing quarterly employment metrics data structure")
        
        new_columns = []
        current_industry = None
        
        for i, col in enumerate(columns):
            if isinstance(col, tuple) and len(col) >= 4:
                # QEM structure: (Title, Industry, Sex, Earnings_Type)
                title = str(col[0]).strip() if len(col) > 0 else ""
                industry_part = str(col[1]).strip() if len(col) > 1 else ""
                sex_part = str(col[2]).strip() if len(col) > 2 else ""
                earnings_part = str(col[3]).strip() if len(col) > 3 else ""
                
                # Check if this column defines a new industry
                if industry_part and industry_part not in ['', 'nan', 'NaN'] and 'Unnamed' not in industry_part:
                    current_industry = industry_part
                
                # Build column name
                col_parts = []
                
                # Add industry (use current industry if available)
                if current_industry:
                    # Clean industry name
                    clean_industry = current_industry.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '').replace('/', '_').replace('&', 'and')
                    # Shorten common long industry names
                    if 'Transport' in clean_industry and 'Postal' in clean_industry and 'Warehousing' in clean_industry:
                        clean_industry = 'Transport_Postal_Warehousing'
                    elif 'Professional' in clean_industry and 'Scientific' in clean_industry:
                        clean_industry = 'Professional_Scientific_Technical'
                    elif 'Accommodation' in clean_industry and 'Food' in clean_industry:
                        clean_industry = 'Accommodation_Food_Services'
                    elif 'Information' in clean_industry and 'Media' in clean_industry:
                        clean_industry = 'Information_Media_Telecommunications'
                    elif 'Financial' in clean_industry and 'Insurance' in clean_industry:
                        clean_industry = 'Financial_Insurance_Services'
                    elif 'Rental' in clean_industry and 'Hiring' in clean_industry:
                        clean_industry = 'Rental_Hiring_Real_Estate'
                    elif 'Health' in clean_industry and 'Care' in clean_industry:
                        clean_industry = 'Health_Care_Social_Assistance'
                    elif 'Arts' in clean_industry and 'Recreation' in clean_industry:
                        clean_industry = 'Arts_Recreation_Other_Services'
                    elif 'Public' in clean_industry and 'Administration' in clean_industry:
                        clean_industry = 'Public_Administration_Safety'
                    elif 'Education' in clean_industry and 'Training' in clean_industry:
                        clean_industry = 'Education_Training'
                    elif 'Total' in clean_industry and 'All' in clean_industry:
                        clean_industry = 'All_Industries'
                    elif 'Electricity' in clean_industry and 'Gas' in clean_industry:
                        clean_industry = 'Electricity_Gas_Water_Waste'
                    elif 'Forestry' in clean_industry and 'Mining' in clean_industry:
                        clean_industry = 'Forestry_Mining'
                    
                    col_parts.append(clean_industry)
                
                # Add sex category
                if sex_part and sex_part not in ['', 'nan', 'NaN'] and 'Unnamed' not in sex_part:
                    clean_sex = sex_part.replace(' ', '_').replace('Both', 'Total')
                    if 'Total_Both_Sexes' in clean_sex:
                        clean_sex = 'Total_Both_Sexes'
                    col_parts.append(clean_sex)
                
                # Add earnings/employment type
                if earnings_part and earnings_part not in ['', 'nan', 'NaN'] and 'Unnamed' not in earnings_part:
                    clean_earnings = earnings_part.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_Plus_')
                    # Shorten long earnings type names
                    if 'Ordinary_Time_Hourly' in clean_earnings:
                        clean_earnings = 'OT_Hourly'
                    elif 'Overtime_Hourly' in clean_earnings:
                        clean_earnings = 'Overtime_Hourly'
                    elif 'Total_Ordinary_Time_Plus_Overtime_Hourly' in clean_earnings:
                        clean_earnings = 'Total_Hourly'
                    elif 'Total' in clean_earnings and 'Hourly' in clean_earnings:
                        clean_earnings = 'Total_Hourly'
                    # Handle employment status types
                    elif 'Part_Time_Paid_Employee' in clean_earnings:
                        clean_earnings = 'Part_Time_Employee'
                    elif 'Full_Time_Paid_Employee' in clean_earnings:
                        clean_earnings = 'Full_Time_Employee'
                    elif 'Total_Status_in_Employment' in clean_earnings:
                        clean_earnings = 'Total_Employment'
                    elif 'Self_Employed' in clean_earnings:
                        clean_earnings = 'Self_Employed'
                    elif 'Working_Proprietor' in clean_earnings:
                        clean_earnings = 'Working_Proprietor'
                    
                    col_parts.append(clean_earnings)
                
                # Construct final name
                if col_parts:
                    col_name = '_'.join(col_parts)
                else:
                    # Fallback - use meaningful parts
                    meaningful_parts = []
                    for part in col:
                        part_str = str(part).strip()
                        if (part_str and 
                            'Unnamed' not in part_str and 
                            part_str not in ['', 'nan', 'NaN'] and
                            'level_' not in part_str):
                            meaningful_parts.append(part_str)
                    
                    if meaningful_parts:
                        col_name = '_'.join([p.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '') for p in meaningful_parts])
                    else:
                        col_name = f'QEM_column_{i}'
                
                # Clean the final name
                col_name = re.sub(r'\s+', '_', col_name)
                col_name = re.sub(r'[^\w\s-]', '_', col_name)
                col_name = col_name.replace('-', '_').replace('/', '_')
                col_name = re.sub(r'_+', '_', col_name).strip('_')
                
            else:
                # Handle non-tuple or simple columns (like date column)
                if isinstance(col, tuple):
                    meaningful_parts = []
                    for part in col:
                        part_str = str(part).strip()
                        if (part_str and 
                            'Unnamed' not in part_str and 
                            part_str not in ['', 'nan', 'NaN'] and
                            'level_' not in part_str):
                            meaningful_parts.append(part_str)
                    
                    if meaningful_parts:
                        col_name = '_'.join([p.replace(' ', '_').replace('-', '_') for p in meaningful_parts])
                    else:
                        col_name = f'column_{i}'
                else:
                    col_name = str(col).strip()
                
                # Clean the name
                if col_name and col_name not in ['', 'nan', 'NaN']:
                    col_name = re.sub(r'\s+', '_', col_name)
                    col_name = re.sub(r'[^\w\s-]', '_', col_name)
                    col_name = col_name.replace('-', '_').replace('/', '_')
                    col_name = re.sub(r'_+', '_', col_name).strip('_')
                else:
                    col_name = f'column_{i}'
            
            new_columns.append(col_name)
        
        # Log the earnings/employment columns found
        qem_cols = [col for col in new_columns if any(word in col.lower() for word in ['hourly', 'earnings', 'ordinary', 'overtime', 'total', 'ot_', 'employee', 'employment', 'part_time', 'full_time'])]
        if qem_cols:
            print(f"    Found {len(qem_cols)} QEM columns")
            print(f"    Sample: {qem_cols[:5]}")
        
        return new_columns
    
    def process_hlf_columns(self, columns):
        """Process HLF 4-level headers: Title/Age_Group/Region/Labour_Force_Metric"""
        print("  HLF: Processing labour force statistics data structure")
        
        new_columns = []
        current_age_group = None
        current_region = None
        
        for i, col in enumerate(columns):
            if isinstance(col, tuple) and len(col) >= 4:
                # HLF structure: (Title, Age_Group, Region, Labour_Metric)
                title = str(col[0]).strip() if len(col) > 0 else ""
                age_group = str(col[1]).strip() if len(col) > 1 else ""
                region = str(col[2]).strip() if len(col) > 2 else ""
                metric = str(col[3]).strip() if len(col) > 3 else ""
                
                # Track current age group (resets when a new age group appears)
                if age_group and age_group not in ['', 'nan', 'NaN', 'Unnamed'] and 'level' not in age_group:
                    current_age_group = age_group
                    # When age group changes, region also resets
                
                # Track current region (resets when a new region appears within same age group)
                if region and region not in ['', 'nan', 'NaN', 'Unnamed'] and 'level' not in region:
                    current_region = region
                
                # Build column name using tracked values
                col_parts = []
                
                # Add age group
                if current_age_group:
                    clean_age = current_age_group.replace(' ', '_').replace('-', '_to_')
                    if '15-24' in clean_age:
                        clean_age = '15_to_24_Years'
                    elif '25-54' in clean_age:
                        clean_age = '25_to_54_Years'
                    elif '55' in clean_age and 'Over' in clean_age:
                        clean_age = '55_Plus_Years'
                    elif 'Total' in clean_age or 'All Ages' in clean_age:
                        clean_age = 'All_Ages'
                    col_parts.append(clean_age)
                
                # Add region
                if current_region:
                    clean_region = current_region.replace(' ', '_')
                    col_parts.append(clean_region)
                
                # Add labour force metric
                if metric and metric not in ['', 'nan', 'NaN', 'Unnamed']:
                    clean_metric = metric.replace(' ', '_').replace('/', '_')
                    # Standardize common metric names
                    if 'Persons Employed' in metric:
                        clean_metric = 'Employed_Persons'
                    elif 'Persons Unemployed' in metric:
                        clean_metric = 'Unemployed_Persons'
                    elif 'Not in Labour Force' in metric:
                        clean_metric = 'Not_in_Labour_Force'
                    elif 'Working Age Population' in metric:
                        clean_metric = 'Working_Age_Population'
                    elif 'Labour Force Participation Rate' in metric:
                        clean_metric = 'Labour_Force_Participation_Rate'
                    elif 'Unemployment Rate' in metric:
                        clean_metric = 'Unemployment_Rate'
                    elif 'Employment Rate' in metric:
                        clean_metric = 'Employment_Rate'
                    elif 'Total Labour Force' in metric:
                        clean_metric = 'Total_Labour_Force'
                    col_parts.append(clean_metric)
                
                # Construct final name
                if col_parts:
                    col_name = '_'.join(col_parts)
                else:
                    # Fallback - use meaningful parts
                    meaningful_parts = []
                    for part in col:
                        part_str = str(part).strip()
                        if (part_str and 
                            'Unnamed' not in part_str and 
                            part_str not in ['', 'nan', 'NaN']):
                            meaningful_parts.append(part_str)
                    
                    if meaningful_parts:
                        col_name = '_'.join([p.replace(' ', '_').replace('-', '_') for p in meaningful_parts])
                    else:
                        col_name = f'HLF_column_{i}'
                
                # Clean the final name
                col_name = re.sub(r'\s+', '_', col_name)
                col_name = re.sub(r'[^\w\s-]', '_', col_name)
                col_name = col_name.replace('-', '_').replace('/', '_')
                col_name = re.sub(r'_+', '_', col_name).strip('_')
                
            else:
                # Handle non-tuple or simple columns (like date column)
                if isinstance(col, tuple):
                    meaningful_parts = []
                    for part in col:
                        part_str = str(part).strip()
                        if (part_str and 
                            'Unnamed' not in part_str and 
                            part_str not in ['', 'nan', 'NaN'] and
                            'level_' not in part_str):
                            meaningful_parts.append(part_str)
                    
                    if meaningful_parts:
                        col_name = '_'.join([p.replace(' ', '_').replace('-', '_') for p in meaningful_parts])
                    else:
                        col_name = f'column_{i}'
                else:
                    col_name = str(col).strip()
                
                col_name = re.sub(r'\s+', '_', col_name)
                col_name = re.sub(r'[^\w\s-]', '_', col_name)
                col_name = col_name.replace('-', '_').replace('/', '_')
                col_name = re.sub(r'_+', '_', col_name).strip('_')
                
                if not col_name or col_name in ['nan', 'NaN', '..']:
                    col_name = f'column_{i}'
            
            new_columns.append(col_name)
        
        # Log the labour force columns found
        labour_cols = [col for col in new_columns if any(term in col for term in ['Employed', 'Unemployed', 'Labour_Force', 'Employment_Rate', 'Participation'])]
        if labour_cols:
            print(f"    Found {len(labour_cols)} HLF labour force columns")
            print(f"    Sample: {labour_cols[:5]}")
        
        return new_columns
    
    def detect_and_fix_date_column(self, df):
        """Detect and standardize date column"""
        date_column = None
        
        # FIRST: Check first column content for date patterns (most reliable for quarterly data)
        if len(df.columns) > 0:
            first_col = df.columns[0]
            sample_values = df[first_col].dropna().head(5)
            date_like_count = 0
            
            for val in sample_values:
                val_str = str(val)
                # Check for date patterns (quarterly like 2009Q3, monthly like 2000M01)
                if (any(char in val_str for char in ['-', '/', 'Q', 'M']) or 
                    (val_str.isdigit() and len(val_str) == 4) or  # Year
                    ('Q' in val_str and any(c.isdigit() for c in val_str)) or  # Quarter
                    ('M' in val_str and any(c.isdigit() for c in val_str))):  # Monthly
                    date_like_count += 1
                    
            if date_like_count >= 3:  # Most samples look date-like
                date_column = first_col
        
        # SECOND: Only if first column doesn't contain dates, look for date column names
        if not date_column:
            date_keywords = ['date', 'quarter', 'month', 'year', 'period', 'time']
            for col in df.columns:
                col_lower = str(col).lower()
                # Be more selective - avoid matching "age" columns
                if any(keyword in col_lower for keyword in date_keywords) and 'age' not in col_lower:
                    date_column = col
                    break
        
        # Rename to standard 'date' column
        if date_column and date_column != 'date':
            df = df.rename(columns={date_column: 'date'})
            
        return df
    
    def clean_csv_file(self, filename):
        """Clean individual CSV with robust government data handling"""
        filepath = self.source_dir / filename
        if not filepath.exists():
            self.log_action("SKIP", f"File not found: {filename}")
            return None
            
        print(f"Processing: {filename}")
        
        try:
            # Detect optimal header structure
            config = self.detect_header_structure(filepath)
            if config["quality"] == 0:
                print(f"  WARNING: Could not parse {filename}")
                return None
            
            # Load with detected configuration
            if config["header_rows"] == 0:
                df = pd.read_csv(filepath)
            elif config.get("is_mei", False):
                # Special MEI file loading - check if industry-based (4 levels) or age/region-based (5 levels)
                if config.get("is_industry_mei", False):
                    df = pd.read_csv(filepath, header=[0, 1, 2, 3])
                else:
                    df = pd.read_csv(filepath, header=[0, 1, 2, 3, 4])
            elif config.get("is_ect", False):
                # Special ECT file loading with dynamic header levels (3 or 4)
                header_levels = config.get("header_levels", 3)
                if header_levels == 4:
                    df = pd.read_csv(filepath, header=[0, 1, 2, 3])
                else:
                    df = pd.read_csv(filepath, header=[0, 1, 2])
            elif config.get("is_qem_sector", False):
                # Special QEM sector file loading with 5-level headers
                df = pd.read_csv(filepath, header=[0, 1, 2, 3, 4])
            elif config.get("is_qem", False):
                # Special QEM file loading with 4-level headers
                df = pd.read_csv(filepath, header=[0, 1, 2, 3])
            elif config.get("is_hlf", False):
                # Special HLF file loading with 4-level headers
                df = pd.read_csv(filepath, header=[0, 1, 2, 3])
            elif config.get("is_lci", False):
                # Special LCI file loading with 3-level headers
                df = pd.read_csv(filepath, header=[0, 1, 2])
            else:
                df = pd.read_csv(filepath, header=list(range(config["header_rows"])))
            
            if df.empty:
                print(f"  SKIP: Empty dataset")
                return None
            
            # Standardize columns
            df = self.standardize_columns(df)
            
            # Fix date column
            df = self.detect_and_fix_date_column(df)
            
            # Basic cleaning - be more conservative about column removal
            df = df.dropna(how='all')  # Remove empty rows only
            df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicate columns
            
            # Only remove columns that are completely empty (all NaN, all empty strings, or all "..")
            def is_truly_empty_column(col):
                """Check if column is truly empty (all NaN, empty strings, or government missing markers)"""
                if col.isna().all():
                    return True
                # Convert to string and check for empty/missing patterns
                str_col = col.astype(str).str.strip()
                non_empty = str_col[(str_col != '') & (str_col != 'nan') & (str_col != '..') & (str_col != 'NaN')]
                return len(non_empty) == 0
            
            # Only drop truly empty columns
            cols_to_drop = [col for col in df.columns if is_truly_empty_column(df[col])]
            if cols_to_drop:
                print(f"    Removing {len(cols_to_drop)} truly empty columns")
                df = df.drop(columns=cols_to_drop)
            
            # Save cleaned dataset
            output_path = self.output_dir / f"cleaned_{filename}"
            df.to_csv(output_path, index=False)
            
            print(f"  SUCCESS: {len(df)} rows, {len(df.columns)} columns -> {output_path.name}")
            self.log_action("CLEAN", f"Successfully cleaned {filename}: {len(df)} rows, {len(df.columns)} cols")
            
            return df
            
        except Exception as e:
            print(f"  ERROR: {e}")
            self.log_action("ERROR", f"Failed to clean {filename}: {e}")
            return None
    
    def generate_audit_report(self):
        """Generate simple audit report"""
        report_path = self.output_dir / "audit_log.json"
        with open(report_path, 'w') as f:
            json.dump(self.audit_log, f, indent=2)
        print(f"Audit report saved: {report_path}")
    
    def process_all_files(self):
        """Process all CSV files in source directory"""
        print("="*60)
        print("COMPREHENSIVE DATA CLEANING - ESSENTIAL VERSION")
        print("="*60)
        
        if not self.source_dir.exists():
            print(f"ERROR: Source directory not found: {self.source_dir}")
            return []
        
        csv_files = list(self.source_dir.glob("*.csv"))
        if not csv_files:
            print(f"ERROR: No CSV files found in {self.source_dir}")
            return []
        
        print(f"Processing {len(csv_files)} CSV files...")
        
        cleaned_datasets = []
        for csv_file in csv_files:
            result = self.clean_csv_file(csv_file.name)
            if result is not None:
                cleaned_datasets.append(result)
        
        # Generate audit report
        self.generate_audit_report()
        
        print(f"\nData cleaning completed: {len(cleaned_datasets)} files processed")
        return cleaned_datasets

def main():
    cleaner = GovernmentDataCleaner()
    cleaned_datasets = cleaner.process_all_files()
    return len(cleaned_datasets) > 0

if __name__ == "__main__":
    main()