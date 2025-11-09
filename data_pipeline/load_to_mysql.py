#!/usr/bin/env python3
"""
MySQL Data Loader
Loads processed train/test datasets into MySQL database
"""

import pandas as pd
import mysql.connector
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
DATA_DIR = Path(__file__).parent / "processed_data"

def get_db_connection():
    """Create MySQL database connection"""
    db_url = os.getenv('DATABASE_URL', 'mysql://root:password@localhost:3306/mlops_db')
    
    # Parse DATABASE_URL
    # Format: mysql://user:password@host:port/database
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

def create_tables(cursor):
    """Create tables for COVID and Churn datasets"""
    print("📋 Creating database tables...")
    
    # COVID train table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS covid_train (
            id INT AUTO_INCREMENT PRIMARY KEY,
            country_name VARCHAR(255),
            confirmed_cases INT,
            deaths INT,
            recovered INT,
            active_cases INT,
            death_rate FLOAT,
            recovery_rate FLOAT,
            risk_score FLOAT,
            risk_level VARCHAR(50),
            timestamp DATETIME,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # COVID test table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS covid_test (
            id INT AUTO_INCREMENT PRIMARY KEY,
            country_name VARCHAR(255),
            confirmed_cases INT,
            deaths INT,
            recovered INT,
            active_cases INT,
            death_rate FLOAT,
            recovery_rate FLOAT,
            risk_score FLOAT,
            risk_level VARCHAR(50),
            timestamp DATETIME,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Churn train table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS churn_train (
            id INT AUTO_INCREMENT PRIMARY KEY,
            customer_id VARCHAR(255),
            SeniorCitizen INT,
            Partner INT,
            Dependents INT,
            tenure INT,
            PhoneService INT,
            MultipleLines INT,
            OnlineSecurity INT,
            OnlineBackup INT,
            DeviceProtection INT,
            TechSupport INT,
            StreamingTV INT,
            StreamingMovies INT,
            PaperlessBilling INT,
            MonthlyCharges FLOAT,
            TotalCharges FLOAT,
            gender_encoded INT,
            Contract_encoded INT,
            PaymentMethod_encoded INT,
            InternetService_encoded INT,
            churn INT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Churn test table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS churn_test (
            id INT AUTO_INCREMENT PRIMARY KEY,
            customer_id VARCHAR(255),
            SeniorCitizen INT,
            Partner INT,
            Dependents INT,
            tenure INT,
            PhoneService INT,
            MultipleLines INT,
            OnlineSecurity INT,
            OnlineBackup INT,
            DeviceProtection INT,
            TechSupport INT,
            StreamingTV INT,
            StreamingMovies INT,
            PaperlessBilling INT,
            MonthlyCharges FLOAT,
            TotalCharges FLOAT,
            gender_encoded INT,
            Contract_encoded INT,
            PaymentMethod_encoded INT,
            InternetService_encoded INT,
            churn INT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    print("✅ Tables created successfully")

def load_covid_data(cursor, conn):
    """Load COVID train and test data"""
    print("📥 Loading COVID data...")
    
    train_path = DATA_DIR / "covid_train.csv"
    test_path = DATA_DIR / "covid_test.csv"
    
    if not train_path.exists() or not test_path.exists():
        print("❌ COVID data files not found. Run preprocessing first.")
        return
    
    # Load train data
    train_df = pd.read_csv(train_path)
    for _, row in train_df.iterrows():
        cursor.execute("""
            INSERT INTO covid_train 
            (country_name, confirmed_cases, deaths, recovered, active_cases, 
             death_rate, recovery_rate, risk_score, risk_level, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            row.get('country_name'),
            int(row.get('confirmed_cases', 0)),
            int(row.get('deaths', 0)),
            int(row.get('recovered', 0)),
            int(row.get('active_cases', 0)),
            float(row.get('death_rate', 0)),
            float(row.get('recovery_rate', 0)),
            float(row.get('risk_score', 0)),
            row.get('risk_level'),
            row.get('timestamp')
        ))
    
    # Load test data
    test_df = pd.read_csv(test_path)
    for _, row in test_df.iterrows():
        cursor.execute("""
            INSERT INTO covid_test 
            (country_name, confirmed_cases, deaths, recovered, active_cases, 
             death_rate, recovery_rate, risk_score, risk_level, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            row.get('country_name'),
            int(row.get('confirmed_cases', 0)),
            int(row.get('deaths', 0)),
            int(row.get('recovered', 0)),
            int(row.get('active_cases', 0)),
            float(row.get('death_rate', 0)),
            float(row.get('recovery_rate', 0)),
            float(row.get('risk_score', 0)),
            row.get('risk_level'),
            row.get('timestamp')
        ))
    
    conn.commit()
    print(f"✅ Loaded {len(train_df)} train and {len(test_df)} test COVID records")

def load_churn_data(cursor, conn):
    """Load Churn train and test data"""
    print("📥 Loading Churn data...")
    
    train_path = DATA_DIR / "churn_train.csv"
    test_path = DATA_DIR / "churn_test.csv"
    
    if not train_path.exists() or not test_path.exists():
        print("❌ Churn data files not found. Run preprocessing first.")
        return
    
    # Load train data
    train_df = pd.read_csv(train_path)
    for _, row in train_df.iterrows():
        cursor.execute("""
            INSERT INTO churn_train 
            (customer_id, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
             MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
             StreamingTV, StreamingMovies, PaperlessBilling, MonthlyCharges, TotalCharges,
             gender_encoded, Contract_encoded, PaymentMethod_encoded, InternetService_encoded, churn)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, tuple(row.get(col, 0) for col in [
            'customer_id', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
            'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
            'gender_encoded', 'Contract_encoded', 'PaymentMethod_encoded', 'InternetService_encoded', 'churn'
        ]))
    
    # Load test data
    test_df = pd.read_csv(test_path)
    for _, row in test_df.iterrows():
        cursor.execute("""
            INSERT INTO churn_test 
            (customer_id, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
             MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
             StreamingTV, StreamingMovies, PaperlessBilling, MonthlyCharges, TotalCharges,
             gender_encoded, Contract_encoded, PaymentMethod_encoded, InternetService_encoded, churn)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, tuple(row.get(col, 0) for col in [
            'customer_id', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
            'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
            'gender_encoded', 'Contract_encoded', 'PaymentMethod_encoded', 'InternetService_encoded', 'churn'
        ]))
    
    conn.commit()
    print(f"✅ Loaded {len(train_df)} train and {len(test_df)} test Churn records")

if __name__ == "__main__":
    print("=" * 60)
    print("MYSQL DATA LOADER")
    print("=" * 60)
    
    try:
        # Connect to database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create tables
        create_tables(cursor)
        
        # Load data
        load_covid_data(cursor, conn)
        load_churn_data(cursor, conn)
        
        # Close connection
        cursor.close()
        conn.close()
        
        print("\n" + "=" * 60)
        print("DATA LOADING COMPLETE")
        print("=" * 60)
        print("✅ All data loaded successfully into MySQL")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
