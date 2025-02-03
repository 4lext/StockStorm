import json
import os
from datetime import datetime
import pandas as pd
from PIL import Image

class FinancialDataStorage:
    def __init__(self, base_path='data/stock_data'):
        self.base_path = base_path
        os.makedirs(base_path,exist_ok=True)
    
    def _get_filepath(self, symbol):
        return f"{self.base_path}/{symbol}.json"
    
    def save_stock_data(self, symbol, data, metadata=None):
        filepath = self._get_filepath(symbol)
        structured_data = {
            'metadata': {
                'symbol': symbol,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'source': 'Alpaca',
                **metadata
            },
            'raw_data': data.to_dict(orient='records'),
            'indicators': ['RSI', 'MACD', 'Bollinger'],
            'patterns': list(data.filter(regex='_STAR|DOJI|HAMMER'))
        }
        
        with open(filepath, 'w') as f:
            json.dump(structured_data, f, indent=2)
    
    def load_stock_data(self, symbol):
        filepath = self._get_filepath(symbol)
        if not os.path.exists(filepath):
            return None
            
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Convert back to DataFrame
        df = pd.DataFrame(data['raw_data'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df, data['metadata']
    
    def update_metadata(self, symbol, new_metadata):
        filepath = self._get_filepath(symbol)
        if os.path.exists(filepath):
            with open(filepath, 'r+') as f:
                data = json.load(f)
                data['metadata'].update(new_metadata)
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
    
    def save_chart(self, symbol, chart_path):
        """Save chart metadata with technical indicators context"""
        if not chart_path:
            return

        metadata = {
            'path': chart_path,
            'timestamp': datetime.now().isoformat(),
            'file_size': os.path.getsize(chart_path),
            'resolution': self._get_image_resolution(chart_path)
        }
        
        filepath = self._get_filepath(symbol)
        if os.path.exists(filepath):
            with open(filepath, 'r+') as f:
                data = json.load(f)
                charts = data['metadata'].get('charts', [])
                charts.append(metadata)
                data['metadata']['charts'] = charts[-5:]  # Keep only last 5
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()

    def _get_image_resolution(self, path):
        try:
            with Image.open(path) as img:
                return f"{img.width}x{img.height}"
        except Exception as e:
            print(f"Error reading image metadata: {str(e)}")
            return "Unknown" 