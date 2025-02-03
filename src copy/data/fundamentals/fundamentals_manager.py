"""
Handles fundamental data acquisition, storage, and visualization
"""
import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from src.data.storage.data_manager import FinancialDataStorage

class FundamentalAnalyzer:
    def __init__(self, api_key):
        self.base_url = "https://api.financialdatasets.ai/financials/"
        self.session = requests.Session()
        self.session.headers.update({
            "X-API-KEY": api_key,
            "Accept": "application/json"
        })
        self.storage = FinancialDataStorage(base_path='data/fundamental_data')
    
    def get_fundamentals(self, symbol, period='quarterly', limit=4):
        """Retrieve and store fundamental data"""
        try:
            url = f"{self.base_url}income-statements"
            params = {
                "ticker": symbol,
                "period": period,
                "limit": limit
            }
            
            response = self.session.get(url, params=params)
            if response.status_code != 200:
                return pd.DataFrame()
                
            data = response.json().get('income_statements', [])
            if not data:
                return pd.DataFrame()
            
            # Process and store data
            df = self._process_fundamentals(data)
            self.storage.save_stock_data(symbol, df, {
                'analysis_date': datetime.now().isoformat(),
                'period': period
            })
            return df
            
        except Exception as e:
            print(f"Fundamental data error: {e}")
            return pd.DataFrame()
    
    def _process_fundamentals(self, raw_data):
        """Convert API response to structured DataFrame"""
        metrics = []
        for item in raw_data:
            processed = {
                'report_date': item.get('report_period'),
                'revenue': float(item.get('revenue', 0)) if item.get('revenue') else 0.0,
                'eps': float(item.get('earnings_per_share', 0)) if item.get('earnings_per_share') else 0.0,
                'gross_profit': float(item.get('gross_profit', 0)) if item.get('gross_profit') else 0.0,
                'operating_income': float(item.get('operating_income', 0)) if item.get('operating_income') else 0.0,
                'net_income': float(item.get('net_income', 0)) if item.get('net_income') else 0.0,
                'price': float(item.get('price', 0)) if item.get('price') else 0.0
            }
            
            # Calculate derived metrics
            metrics.append(self._calculate_derived_metrics(processed))
        
        # Create DataFrame and ensure proper date sorting
        df = pd.DataFrame(metrics)
        
        # Convert to datetime and sort
        df['report_date'] = pd.to_datetime(df['report_date'])
        df = df.sort_values('report_date', ascending=True)  # Oldest first
        
        # Calculate proper revenue growth now that we're sorted
        df['revenue_growth_pct'] = df['revenue'].pct_change() * 100
        
        # After sorting and calculating growth:
        df['report_date'] = df['report_date'].dt.strftime('%Y-%m-%d')  # Convert to string
        
        return df
    
    def _calculate_derived_metrics(self, data):
        """Calculate growth rates and ratios"""
        # Calculate PE ratio
        data['pe_ratio'] = data['price'] / data['eps'] if data['eps'] > 0 else 0
        
        return data
    
    def plot_fundamentals(self, df, symbol):
        """Generate separate fundamentals visualization"""
        if df.empty:
            return None
            
        # Convert string dates back to datetime for plotting
        df['report_date'] = pd.to_datetime(df['report_date'])
        
        # Create index-based x-axis values
        df = df.reset_index(drop=True)
        x_axis = np.arange(len(df))
        
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Fundamental Analysis - {symbol}')
        
        # Revenue Growth
        axs[0,0].plot(x_axis, df['revenue'], marker='o', color='blue')
        axs[0,0].set_xticks(x_axis)
        axs[0,0].set_xticklabels([d.strftime('%Y-%m') for d in df['report_date']], rotation=45)
        axs[0,0].set_title('Revenue Trend')
        axs[0,0].set_ylabel('Revenue (M)')
        
        # EPS Growth
        axs[0,1].bar(x_axis, df['eps'], color='green')
        axs[0,1].set_xticks(x_axis)
        axs[0,1].set_xticklabels([d.strftime('%Y-%m') for d in df['report_date']], rotation=45)
        axs[0,1].set_title('Earnings Per Share')
        axs[0,1].set_ylabel('EPS')
        
        # Profit Margins
        df['gross_margin'] = (df['gross_profit'] / df['revenue']) * 100
        df['net_margin'] = (df['net_income'] / df['revenue']) * 100
        axs[1,0].plot(x_axis, df['gross_margin'], label='Gross Margin', marker='o')
        axs[1,0].plot(x_axis, df['net_margin'], label='Net Margin', marker='o')
        axs[1,0].set_xticks(x_axis)
        axs[1,0].set_xticklabels([d.strftime('%Y-%m') for d in df['report_date']], rotation=45)
        axs[1,0].set_title('Profit Margins')
        axs[1,0].set_ylabel('Percentage (%)')
        axs[1,0].legend()
        
        # PE Ratio
        axs[1,1].bar(x_axis, df['pe_ratio'], color='purple')
        axs[1,1].set_xticks(x_axis)
        axs[1,1].set_xticklabels([d.strftime('%Y-%m') for d in df['report_date']], rotation=45)
        axs[1,1].set_title('Price-Earnings Ratio')
        axs[1,1].set_ylabel('PE Ratio')
        
        plt.tight_layout()
        
        # Save chart
        chart_dir = os.path.join("charts/fundamental_charts", symbol, datetime.now().strftime('%Y-%m'))
        os.makedirs(chart_dir, exist_ok=True)
        chart_path = os.path.join(chart_dir, f"{symbol}_fundamentals_{datetime.now().strftime('%Y%m%d')}.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        return chart_path 