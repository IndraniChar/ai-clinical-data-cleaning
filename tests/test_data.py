# tests/test_data.py
import pytest
import pandas as pd
import sqlite3
import os
import numpy as np

# Set test environment variables before importing app
os.environ["ADMIN_USER"] = "test_admin"
os.environ["ADMIN_PASS"] = "test_password"

# Now import app components
from app import analyze_data, init_db, save_to_db

@pytest.fixture(scope="module")
def test_db():
    """Test database setup with clean state"""
    # Initialize fresh test database
    init_db()
    yield  # Test runs happen here
    
    # Teardown - remove test database
    try:
        os.remove("clinical_data.db")
    except FileNotFoundError:
        pass

@pytest.fixture
def good_data():
    """Sample valid clinical data"""
    return pd.DataFrame({
        'patient_id': ['PT-1001'],
        'age': [45],
        'systolic_bp': [120],
        'diastolic_bp': [80],
        'gender': ['M']
    })

@pytest.fixture 
def bad_data():
    """Sample data with anomalies"""
    return pd.DataFrame({
        'patient_id': ['PT-1002'],
        'age': [150],  # Invalid age
        'systolic_bp': [220],  # High BP
        'diastolic_bp': [0],  # Low BP
        'gender': ['X']  # Invalid gender
    })

def test_normal_data(good_data):
    """Test clean data passes validation"""
    df, _ = analyze_data(good_data)
    assert not df['is_anomaly'].any()
    assert 'ml_anomaly' in df.columns

def test_anomalies(bad_data):
    """Test anomaly detection"""
    df, types = analyze_data(bad_data)
    assert df['is_anomaly'].any()
    assert {'High SBP', 'Extreme Age'}.issubset(set(types))

def test_missing_cols():
    """Test required column validation"""
    with pytest.raises(ValueError, match="Missing required columns"):
        analyze_data(pd.DataFrame({'invalid_col': [1]}))

def test_db_operations(test_db, good_data):
    """Test database integration"""
    conn = sqlite3.connect('clinical_data.db')
    save_to_db(good_data)
    result = pd.read_sql("SELECT * FROM patients", conn)
    assert len(result) == 1
    assert result.iloc[0]['age'] == 45
    conn.close()

def test_ml_consistency(bad_data):
    """Test ML model consistency"""
    df1, _ = analyze_data(bad_data)
    df2, _ = analyze_data(bad_data)  # Same input should produce same output
    assert np.array_equal(df1['ml_anomaly'], df2['ml_anomaly'])