#!/usr/bin/env python3
"""
Prepare Feast Data
Exports MySQL data to Parquet format for Feast
"""

import pandas as pd
import mysql.connector
from pathlib import Path
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Paths
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

def get_db_connection():
    """Create MySQL database connection"""
    db_url = os.getenv('DATABASE_URL', 'mysql://root:password@localhost:3306/mlops_db')
    
    # Parse DATABASE_URL
    parts = db_url.replace('mysql://', '').split('@')
    user_pass = parts[0].split(':')
    host_db = parts[1].split('/')
    host_port = host_db[0].split(':')
    
    config = {
        'user': user_pass[0],
        'password': user_pass[1],
        'host': host_port[0],
        'port': int(host_port[1]) if len(host_port) > 1 else 3306,
        'database': host_db[1] if len(host_db) > 1 else 'mlops_db'
    }
    
    return mysql.connector.connect(**config)

def export_covid_features():
    """Export COVID features to Parquet"""
    print("📤 Exporting COVID features...")
    
    conn = get_db_connection()
    
    # Query COVID test data (latest records for serving)
    query = """
        SELECT 
            country_name,
            confirmed_cases,
            deaths,
            recovered,
            active_cases,
            death_rate,
            recovery_rate,
            risk_score,
            timestamp,
            created_at as event_timestamp
        FROM covid_test
        ORDER BY created_at DESC
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    # Convert timestamp columns
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
    
    # Save to parquet
    output_path = DATA_DIR / "covid_features.parquet"
    df.to_parquet(output_path, index=False)
    
    print(f"✅ Exported {len(df)} COVID feature records to {output_path}")
    return df

def export_churn_features():
    """Export Churn features to Parquet"""
    print("📤 Exporting Churn features...")
    
    conn = get_db_connection()
    
    # Query Churn test data (latest records for serving)
    query = """
        SELECT 
            customer_id,
            SeniorCitizen,
            Partner,
            Dependents,
            tenure,
            PhoneService,
            MultipleLines,
            OnlineSecurity,
            OnlineBackup,
            DeviceProtection,
            TechSupport,
            StreamingTV,
            StreamingMovies,
            PaperlessBilling,
            MonthlyCharges,
            TotalCharges,
            gender_encoded,
            Contract_encoded,
            PaymentMethod_encoded,
            InternetService_encoded,
            created_at as event_timestamp
        FROM churn_test
        ORDER BY created_at DESC
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    # Convert timestamp column
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
    
    # Save to parquet
    output_path = DATA_DIR / "churn_features.parquet"
    df.to_parquet(output_path, index=False)
    
    print(f"✅ Exported {len(df)} Churn feature records to {output_path}")
    return df

if __name__ == "__main__":
    print("=" * 60)
    print("FEAST DATA PREPARATION")
    print("=" * 60)
    
    try:
        # Export features
        covid_df = export_covid_features()
        churn_df = export_churn_features()
        
        print("\n" + "=" * 60)
        print("EXPORT COMPLETE")
        print("=" * 60)
        print(f"COVID features: {len(covid_df)} records")
        print(f"Churn features: {len(churn_df)} records")
        print("\nNext steps:")
        print("1. cd feast_store")
        print("2. feast apply")
        print("3. feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
