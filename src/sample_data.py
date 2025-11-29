"""
Sample Data Generator
Provides sample telemetry data for testing and demonstration
"""

import pandas as pd


def create_sample_data():
    """
    Generate sample telemetry data for demonstration
    
    Returns:
        DataFrame with sample sensor readings for 3 machines
    """
    sample_data = {
        'machineID': [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3],
        'datetime': ['2015-12-01 06:00:00', '2015-12-01 07:00:00', '2015-12-01 08:00:00', 
                     '2015-12-01 09:00:00', '2015-12-01 10:00:00'] * 3,
        'volt': [176.217,162.877,162.348,175.482,168.953] * 3,
        'rotate': [418.504,402.604,407.391,428.762,412.318] * 3,
        'pressure': [113.048,95.016,103.948,108.234,99.567] * 3,
        'vibration': [40.365,40.405,40.056,39.892,40.128] * 3,
        'model': ['model1']*5 + ['model2']*5 + ['model3']*5,
        'age': [18]*5 + [7]*5 + [4]*5
    }
    return pd.DataFrame(sample_data)
