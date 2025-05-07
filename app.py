# app.py - UPDATED VERSION WITH FIXES
import os
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sqlite3
from fpdf import FPDF
from sklearn.ensemble import IsolationForest
import streamlit_authenticator as stauth
from dotenv import load_dotenv
import extra_streamlit_components as stx

# --- Constants ---
REQUIRED_COLUMNS = {
    'patient_id': 'Unique patient identifier',
    'age': 'Numeric (18-120 expected)',
    'systolic_bp': 'Numeric blood pressure',
    'diastolic_bp': 'Numeric blood pressure'
}

# --- Database Setup ---
def init_db():
    """Initialize SQLite database with proper schema"""
    conn = sqlite3.connect('clinical_data.db')
    c = conn.cursor()
    
    # Create patients table with all required columns
    c.execute('''CREATE TABLE IF NOT EXISTS patients
                 (patient_id TEXT,
                  age REAL,
                  systolic_bp REAL,
                  diastolic_bp REAL,
                  ml_anomaly INTEGER DEFAULT 0,
                  is_anomaly INTEGER DEFAULT 0)''')
                  
    c.execute('''CREATE TABLE IF NOT EXISTS analyses
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                 patient_count INTEGER,
                 anomaly_count INTEGER,
                 top_issue TEXT)''')
    conn.commit()
    conn.close()

# --- Authentication ---
def setup_auth():
    """Configure secure authentication"""
    if 'auth' not in st.session_state:
        cookie_manager = stx.CookieManager()
        load_dotenv()
        credentials = {
            "usernames": {
                os.getenv("ADMIN_USER", "admin"): {
                    "name": "Admin",
                    "password": stauth.Hasher([os.getenv("ADMIN_PASS", "admin123")]).generate()[0]
                }
            }
        }
        st.session_state.auth = stauth.Authenticate(
            credentials,
            "clinical_auth",
            "auth_key",
            cookie_expiry_days=30,
            cookie_manager=cookie_manager
        )
    return st.session_state.auth.login("Login", "main")

# --- Core Analysis ---
def analyze_data(df, threshold=0.2):
    """Enhanced anomaly detection with ML"""
    try:
        # Validate input
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        df = df.copy()
        numeric_cols = ['age', 'systolic_bp', 'diastolic_bp']
        
        # Convert and clean data
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].median(), inplace=True)
        
        # Rule-based anomalies
        conditions = {
            'High SBP': df['systolic_bp'] > 180,
            'Low DBP': df['diastolic_bp'] < 40,
            'Extreme Age': (df['age'] < 18) | (df['age'] > 120)
        }
        
        # ML Anomaly Detection
        clf = IsolationForest(contamination=threshold, random_state=42)
        df['ml_anomaly'] = (clf.fit_predict(df[numeric_cols]) == -1).astype(int)
        
        # Combine results
        for name, condition in conditions.items():
            df[name] = condition.astype(int)
        df['is_anomaly'] = (df[list(conditions.keys())].any(axis=1) | (df['ml_anomaly'] == 1)).astype(int)
        
        return df, list(conditions.keys())
    
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return pd.DataFrame(), []

# --- Report Generation ---
def generate_report(df, format='pdf'):
    """Create downloadable reports"""
    if format == 'pdf':
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Clinical Data Report", ln=1, align='C')
        pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=1)
        pdf.cell(200, 10, txt=f"Patients Analyzed: {len(df)}", ln=1)
        pdf.cell(200, 10, txt=f"Anomalies Found: {df['is_anomaly'].sum()}", ln=1)
        pdf.output("report.pdf")
        return open("report.pdf", "rb")
    else:
        return df.to_csv(index=False).encode()

# --- Main App ---
def main():
    # Initialize
    init_db()
    name, auth_status, username = setup_auth()
    
    if not auth_status:
        return
    
    st.title(f"🏥 Clinical Data Dashboard | Welcome {name}")
    
    with st.sidebar:
        st.session_state.auth.logout("Logout", "sidebar")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        sensitivity = st.slider("Anomaly Sensitivity", 0.1, 0.5, 0.2, 0.05)
    
    if uploaded_file:
        try:
            # Process data
            raw_df = pd.read_csv(uploaded_file)
            processed_df, anomalies = analyze_data(raw_df, sensitivity)
            
            # Ensure all required columns exist before saving
            for col in ['ml_anomaly', 'is_anomaly'] + anomalies:
                if col not in processed_df.columns:
                    processed_df[col] = 0
            
            # Save to database
            conn = sqlite3.connect('clinical_data.db')
            processed_df.to_sql('patients', conn, if_exists='append', index=False)
            conn.close()
            
            # Display results
            st.success(f"✅ Analyzed {len(processed_df)} records")
            
            # Metrics dashboard
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Records", len(processed_df))
            col2.metric("Anomalies", processed_df['is_anomaly'].sum())
            top_issue = processed_df[anomalies].sum().idxmax() if anomalies else "None"
            col3.metric("Top Issue", top_issue)
            
            # Tabs interface
            tab1, tab2, tab3 = st.tabs(["📊 Data", "📈 Visuals", "📥 Export"])
            
            with tab1:
                st.dataframe(processed_df)
            
            with tab2:
                fig, ax = plt.subplots()
                sns.scatterplot(
                    data=processed_df,
                    x='age',
                    y='systolic_bp',
                    hue='is_anomaly',
                    palette={1: 'red', 0: 'green'}
                )
                st.pyplot(fig)
            
            with tab3:
                export_format = st.radio("Format", ["PDF", "CSV"])
                if st.button("Generate Report"):
                    report = generate_report(processed_df, export_format.lower())
                    st.download_button(
                        f"Download {export_format}",
                        report,
                        f"clinical_report_{datetime.now().date()}.{export_format.lower()}"
                    )
        
        except Exception as e:
            st.error(f"Processing error: {str(e)}")

if __name__ == "__main__":
    main()