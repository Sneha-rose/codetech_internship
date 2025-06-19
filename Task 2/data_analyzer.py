import os
import csv
import json
import xml.etree.ElementTree as ET
import pandas as pd
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import tempfile

# Set Seaborn style properly
sns.set_style("whitegrid")

# Prompt user to enter the file path
file_path = input("Enter path to data file: ").strip()

# Check if file exists
if not os.path.exists(file_path):
    print("File not found. Please check the path and try again.")
    exit()

print("Loading data...")
df = pd.read_csv(file_path)

print(f"Loaded {len(df)} records with {len(df.columns)} columns")
print("Analyzing data...")

# Show basic info
print("\nData Summary:")
print(df.describe(include='all'))

class DataAnalyzer:
    def __init__(self, filename):
        self.filename = filename
        self.data = None
        self.file_type = None
        self.analysis_results = {}
        self.chart_files = []
        self.temp_dir = tempfile.mkdtemp()
        
    def detect_file_type(self):
        """Detect file type based on extension"""
        ext = os.path.splitext(self.filename)[1].lower()
        if ext in ['.csv', '.json', '.xml', '.xlsx', '.xls']:
            return ext[1:]
        raise ValueError(f"Unsupported file format: {ext}")

    def load_data(self):
        """Load data from various file formats"""
        self.file_type = self.detect_file_type()
        
        if self.file_type == 'csv':
            self.data = pd.read_csv(self.filename)
        elif self.file_type == 'json':
            self.data = pd.read_json(self.filename)
        elif self.file_type == 'xml':
            tree = ET.parse(self.filename)
            root = tree.getroot()
            data = []
            for child in root:
                record = {}
                for element in child:
                    record[element.tag] = element.text
                data.append(record)
            self.data = pd.DataFrame(data)
        elif self.file_type in ['xlsx', 'xls']:
            self.data = pd.read_excel(self.filename)
        else:
            raise ValueError("Unsupported file format")
        
        # Clean column names
        self.data.columns = [col.strip().replace(' ', '_').lower() for col in self.data.columns]
        return self.data

    def analyze_data(self):
        """Perform comprehensive data analysis"""
        if self.data is None or self.data.empty:
            return None
        
        # Basic information
        self.analysis_results['basic_info'] = {
            'file_name': os.path.basename(self.filename),
            'file_type': self.file_type,
            'records': len(self.data),
            'columns': list(self.data.columns),
            'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Data types
        self.analysis_results['data_types'] = self.data.dtypes.astype(str).to_dict()
        
        # Missing values analysis
        missing_values = self.data.isnull().sum()
        self.analysis_results['missing_values'] = {
            'total': int(missing_values.sum()),
            'by_column': missing_values[missing_values > 0].to_dict(),
            'percentage': round(missing_values.sum() / len(self.data) * 100, 2)
        }
        
        # Descriptive statistics
        self.analysis_results['statistics'] = {}
        for col in self.data.select_dtypes(include=[np.number]).columns:
            self.analysis_results['statistics'][col] = {
                'min': round(self.data[col].min(), 2),
                'max': round(self.data[col].max(), 2),
                'mean': round(self.data[col].mean(), 2),
                'median': round(self.data[col].median(), 2),
                'std_dev': round(self.data[col].std(), 2),
                'q1': round(self.data[col].quantile(0.25), 2),
                'q3': round(self.data[col].quantile(0.75), 2)
            }
        
        # Categorical analysis
        self.analysis_results['categorical'] = {}
        for col in self.data.select_dtypes(include=['object', 'category']).columns:
            counts = self.data[col].value_counts()
            self.analysis_results['categorical'][col] = {
                'unique': int(self.data[col].nunique()),
                'top_values': counts.head(10).to_dict()
            }
        
        # Correlation analysis
        if len(self.data.select_dtypes(include=[np.number]).columns) > 1:
            corr_matrix = self.data.corr(numeric_only=True)
            self.analysis_results['correlations'] = {
                'matrix': corr_matrix.to_dict(),
                'strong_positive': [],
                'strong_negative': []
            }
            
            # Find strong correlations
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:
                        pair = (
                            corr_matrix.columns[i], 
                            corr_matrix.columns[j],
                            round(corr_value, 2)
                        )
                        if corr_value > 0.7:
                            self.analysis_results['correlations']['strong_positive'].append(pair)
                        else:
                            self.analysis_results['correlations']['strong_negative'].append(pair)
        
        return self.analysis_results

    def generate_visualizations(self):
        """Create various data visualizations"""
        sns.set_style('whitegrid')
        
        # Histograms for numerical columns
        num_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.data[col], kde=True)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            chart_path = os.path.join(self.temp_dir, f'{col}_hist.png')
            plt.savefig(chart_path, bbox_inches='tight')
            plt.close()
            self.chart_files.append(chart_path)
        
        # Bar charts for top categorical values
        cat_cols = self.data.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            if self.data[col].nunique() > 50:
                continue  # Skip high-cardinality columns
                
            plt.figure(figsize=(10, 6))
            top_values = self.data[col].value_counts().head(10)
            sns.barplot(x=top_values.values, y=top_values.index, orient='h')
            plt.title(f'Top Values in {col}')
            plt.xlabel('Count')
            plt.ylabel(col)
            chart_path = os.path.join(self.temp_dir, f'{col}_bar.png')
            plt.savefig(chart_path, bbox_inches='tight')
            plt.close()
            self.chart_files.append(chart_path)
        
        # Correlation heatmap
        if len(num_cols) > 1:
            plt.figure(figsize=(12, 8))
            corr = self.data.corr(numeric_only=True)
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', 
                        vmin=-1, vmax=1, linewidths=0.5)
            plt.title('Correlation Matrix')
            chart_path = os.path.join(self.temp_dir, 'correlation_heatmap.png')
            plt.savefig(chart_path, bbox_inches='tight')
            plt.close()
            self.chart_files.append(chart_path)
        
        # Pairplot for small datasets
        if len(num_cols) > 1 and len(self.data) < 1000:
            plt.figure(figsize=(12, 8))
            sns.pairplot(self.data[num_cols].sample(min(500, len(self.data))))
            plt.suptitle('Pairwise Relationships', y=1.02)
            chart_path = os.path.join(self.temp_dir, 'pairplot.png')
            plt.savefig(chart_path, bbox_inches='tight')
            plt.close()
            self.chart_files.append(chart_path)
        
        return self.chart_files

    def generate_pdf_report(self, output_file='data_analysis_report.pdf'):
        """Generate comprehensive PDF report"""
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Title page
        pdf.set_font('Arial', 'B', 24)
        pdf.cell(0, 20, 'DATA ANALYSIS REPORT', 0, 1, 'C')
        pdf.ln(10)
        
        pdf.set_font('Arial', '', 16)
        pdf.cell(0, 10, f'Analysis of: {self.analysis_results["basic_info"]["file_name"]}', 0, 1, 'C')
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 8, f'Generated on: {self.analysis_results["basic_info"]["generated_at"]}', 0, 1, 'C')
        pdf.ln(20)
        
        # Basic info table
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Dataset Overview', 0, 1)
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 12)
        basic_info = self.analysis_results['basic_info']
        info_data = [
            ('File Name', basic_info['file_name']),
            ('File Type', basic_info['file_type'].upper()),
            ('Records', f"{basic_info['records']:,}"),
            ('Columns', ', '.join(basic_info['columns']))
        ]
        
        col_width = 60
        for label, value in info_data:
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(col_width, 8, label)
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 8, value)
            pdf.ln()
        
        # Missing values
        pdf.ln(10)
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Data Quality', 0, 1)
        pdf.ln(5)
        
        missing_info = self.analysis_results['missing_values']
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 8, f"Total Missing Values: {missing_info['total']:,} ({missing_info['percentage']}%)")
        pdf.ln(8)
        
        if missing_info['by_column']:
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(40, 8, "Column")
            pdf.cell(40, 8, "Missing Count")
            pdf.cell(40, 8, "Percentage")
            pdf.ln(8)
            
            pdf.set_font('Arial', '', 12)
            for col, count in missing_info['by_column'].items():
                pct = round(count / len(self.data) * 100, 2)
                pdf.cell(40, 8, col)
                pdf.cell(40, 8, f"{count:,}")
                pdf.cell(40, 8, f"{pct}%")
                pdf.ln(8)
        else:
            pdf.cell(0, 8, "No missing values found in any columns")
            pdf.ln(8)
        
        # Numerical statistics
        if self.analysis_results['statistics']:
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, 'Numerical Analysis', 0, 1)
            pdf.ln(5)
            
            cols = list(self.analysis_results['statistics'].keys())
            for i, col in enumerate(cols):
                if i > 0 and i % 2 == 0:
                    pdf.add_page()
                    pdf.set_font('Arial', 'B', 16)
                    pdf.cell(0, 10, 'Numerical Analysis (Continued)', 0, 1)
                    pdf.ln(5)
                
                stats = self.analysis_results['statistics'][col]
                pdf.set_font('Arial', 'B', 14)
                pdf.cell(0, 10, col, 0, 1)
                pdf.ln(3)
                
                pdf.set_font('Arial', '', 12)
                col_width = 45
                row_height = 8
                
                # Header
                pdf.set_fill_color(200, 220, 255)
                pdf.cell(col_width, row_height, "Statistic", 1, 0, 'C', True)
                pdf.cell(col_width, row_height, "Value", 1, 1, 'C', True)
                
                # Data rows
                for stat, value in stats.items():
                    pdf.cell(col_width, row_height, stat.replace('_', ' ').title(), 1)
                    pdf.cell(col_width, row_height, str(value), 1, 1, 'R')
                
                pdf.ln(10)
                
                # Add chart if available
                chart_path = os.path.join(self.temp_dir, f'{col}_hist.png')
                if os.path.exists(chart_path):
                    pdf.image(chart_path, x=10, w=190)
                    pdf.ln(5)
        
        # Categorical analysis
        if self.analysis_results['categorical']:
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, 'Categorical Analysis', 0, 1)
            pdf.ln(5)
            
            for col, data in self.analysis_results['categorical'].items():
                pdf.set_font('Arial', 'B', 14)
                pdf.cell(0, 10, col, 0, 1)
                pdf.ln(3)
                
                pdf.set_font('Arial', '', 12)
                pdf.cell(0, 8, f"Unique values: {data['unique']}")
                pdf.ln(8)
                
                if data['unique'] > 0:
                    # Table header
                    pdf.set_fill_color(200, 220, 255)
                    pdf.cell(140, 8, "Value", 1, 0, 'C', True)
                    pdf.cell(50, 8, "Count", 1, 1, 'C', True)
                    
                    # Table rows
                    for value, count in data['top_values'].items():
                        pdf.cell(140, 8, str(value)[:80], 1)
                        pdf.cell(50, 8, str(count), 1, 1, 'R')
                    
                    pdf.ln(10)
                    
                    # Add chart if available
                    chart_path = os.path.join(self.temp_dir, f'{col}_bar.png')
                    if os.path.exists(chart_path):
                        pdf.image(chart_path, x=10, w=190)
                        pdf.ln(5)
        
        # Correlation analysis
        if 'correlations' in self.analysis_results and self.analysis_results['correlations']:
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, 'Correlation Analysis', 0, 1)
            pdf.ln(5)
            
            # Strong positive correlations
            if self.analysis_results['correlations']['strong_positive']:
                pdf.set_font('Arial', 'B', 14)
                pdf.cell(0, 10, 'Strong Positive Correlations (>0.7)', 0, 1)
                pdf.ln(3)
                
                pdf.set_font('Arial', '', 12)
                for pair in self.analysis_results['correlations']['strong_positive']:
                    pdf.cell(0, 8, f"{pair[0]} & {pair[1]}: {pair[2]}", 0, 1)
                pdf.ln(5)
            
            # Strong negative correlations
            if self.analysis_results['correlations']['strong_negative']:
                pdf.set_font('Arial', 'B', 14)
                pdf.cell(0, 10, 'Strong Negative Correlations (<-0.7)', 0, 1)
                pdf.ln(3)
                
                pdf.set_font('Arial', '', 12)
                for pair in self.analysis_results['correlations']['strong_negative']:
                    pdf.cell(0, 8, f"{pair[0]} & {pair[1]}: {pair[2]}", 0, 1)
                pdf.ln(5)
            
            # Heatmap
            chart_path = os.path.join(self.temp_dir, 'correlation_heatmap.png')
            if os.path.exists(chart_path):
                pdf.image(chart_path, x=10, w=190)
                pdf.ln(5)
            
            # Pairplot
            chart_path = os.path.join(self.temp_dir, 'pairplot.png')
            if os.path.exists(chart_path):
                pdf.add_page()
                pdf.set_font('Arial', 'B', 16)
                pdf.cell(0, 10, 'Pairwise Relationships', 0, 1)
                pdf.ln(5)
                pdf.image(chart_path, x=10, w=190)
        
        # Save report
        pdf.output(output_file)
        return output_file

    def cleanup(self):
        """Remove temporary chart files"""
        for chart_file in self.chart_files:
            if os.path.exists(chart_file):
                os.remove(chart_file)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

def main():
    """Main function to execute the analysis"""
    try:
        # Get input file
        input_file = input("Enter path to data file: ").strip()
        if not os.path.exists(input_file):
            print(f"Error: File not found - {input_file}")
            return
        
        # Initialize analyzer
        analyzer = DataAnalyzer(input_file)
        
        # Load and analyze data
        print("Loading data...")
        data = analyzer.load_data()
        print(f"Loaded {len(data)} records with {len(data.columns)} columns")
        
        print("Analyzing data...")
        analyzer.analyze_data()
        
        print("Generating visualizations...")
        analyzer.generate_visualizations()
        
        # Generate report
        output_file = 'data_analysis_report.pdf'
        print("Creating PDF report...")
        report_path = analyzer.generate_pdf_report(output_file)
        
        # Cleanup
        analyzer.cleanup()
        
        print(f"Report generated successfully: {report_path}")
        print(f"Analysis completed!")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    main()