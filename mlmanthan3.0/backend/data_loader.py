import pandas as pd
import numpy as np
import io

def load_dataset(file_content: bytes, filename: str) -> pd.DataFrame:
    """Loads dataset from bytes (CSV or Excel)."""
    if filename.endswith('.csv'):
        # Try different encodings for CSV
        try:
            df = pd.read_csv(io.BytesIO(file_content), encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(file_content), encoding='ISO-8859-1')
    elif filename.endswith('.xlsx') or filename.endswith('.xls'):
        df = pd.read_excel(io.BytesIO(file_content))
    else:
        raise ValueError("Unsupported file format. Please upload CSV or Excel.")
    return df

def preprocess_retail_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses Online Retail dataset to customer-level RFM features.
    Simulates a 'Treatment' (Campaign) and 'Outcome' (Next Month Spend) for demonstration.
    """
    # Basic cleaning
    df = df.dropna(subset=['CustomerID'])
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalSpend'] = df['Quantity'] * df['UnitPrice']
    
    # Reference date for Recency (max date in dataset)
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    # Aggregate to Customer Level
    customers = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'count',
        'TotalSpend': 'sum',
        'Country': 'first' # Simplified
    }).rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalSpend': 'Monetary'
    })
    
    # Simulate a Campaign Treatment based on RFM (confounded)
    # High value customers are more likely to be targeted
    np.random.seed(42)
    customers['propensity'] = 1 / (1 + np.exp(-(
        -2 + 
        0.001 * customers['Monetary'] + 
        0.01 * customers['Frequency'] - 
        0.005 * customers['Recency']
    )))
    customers['Treatment'] = np.random.binomial(1, customers['propensity'])
    
    # Simulate Outcome (Next Purchase Amount)
    # Treatment effect = $50 uplift
    customers['Outcome'] = (
        customers['Monetary'] * 0.1 + # Baseline correlation
        customers['Treatment'] * 50 + # True Causal Effect
        np.random.normal(0, 20, size=len(customers)) # Noise
    )
    customers['Outcome'] = customers['Outcome'].clip(lower=0)
    
    return customers.reset_index()

def simulate_dataset(n_samples=1000):
    """Generates synthetic e-commerce data with known causal structure."""
    np.random.seed(42)
    data = pd.DataFrame()
    
    # Confounders
    data['Age'] = np.random.randint(18, 70, n_samples)
    data['Income'] = np.random.normal(50000, 15000, n_samples)
    data['LoyaltyScore'] = np.random.uniform(0, 10, n_samples)
    
    # Treatment Assignment (Campaign Email)
    # Older and higher income people are more likely to get the email
    prob_treatment = 1 / (1 + np.exp(-( -3 + 0.05*data['Age'] + 0.00002*data['Income'] )))
    data['Treatment'] = np.random.binomial(1, prob_treatment)
    
    # Outcome (Purchase Amount)
    # Causal Effect: Treatment adds $20
    # Confounders also affect outcome
    data['Outcome'] = (
        10 + 
        0.5 * data['Age'] + 
        0.001 * data['Income'] + 
        5 * data['LoyaltyScore'] + 
        20 * data['Treatment'] + # True Uplift
        np.random.normal(0, 10, n_samples)
    )
    data['Outcome'] = data['Outcome'].clip(lower=0)
    
    return data
