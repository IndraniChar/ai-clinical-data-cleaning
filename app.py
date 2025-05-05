# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from io import BytesIO
import time
import base64
import sqlite3
from fpdf import FPDF
from sklearn.ensemble import IsolationForest
import streamlit_authenticator as stauth
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="Clinical Data AI Cleaner",
    page_icon="🏥",
    layout="wide"
)

# --- Constants ---
REQUIRED_COLUMNS = {
    'patient_id': 'Unique patient identifier',
    'age': 'Numeric (18-120 expected)',
    'systolic_bp': 'Numeric blood pressure',
    'diastolic_bp': 'Numeric blood pressure'
}

# --- Database Setup ---
def init_db():
    """Initialize SQLite database with all required columns"""
    conn = sqlite3.connect('clinical_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS patients
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 patient_id TEXT, 
                 age REAL, 
                 gender TEXT,
                 systolic_bp REAL, 
                 diastolic_bp REAL,
                 diagnosis TEXT,
                 visit_date TEXT,
                 is_anomaly BOOLEAN,
                 timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

# --- Authentication ---
def setup_auth():
    """Secure authentication setup with environment variables"""
    if 'auth_setup' not in st.session_state:
        # Get credentials from .env or use defaults
        username = os.getenv("ADMIN_USER", "admin")
        password = os.getenv("ADMIN_PASS", "admin123")
        
        # Hash password
        hashed_pass = stauth.Hasher([password]).generate()[0]
        
        credentials = {
            "usernames": {
                username: {
                    "name": "Admin",
                    "password": hashed_pass
                }
            }
        }

        # Initialize authenticator
        st.session_state.authenticator = stauth.Authenticate(
            credentials,
            "clinical_auth_cookie",
            "clinical_auth_key",
            cookie_expiry_days=30
        )
        st.session_state.auth_setup = True
    
    return st.session_state.authenticator.login("Login", "main")

# --- Core Analysis Functions ---
def analyze_data(df, threshold=0.2):
    """Enhanced anomaly detection with ML"""
    # Validation
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    try:
        results = df.copy()
        
        # Convert numeric columns
        numeric_cols = ['age', 'systolic_bp', 'diastolic_bp']
        for col in numeric_cols:
            results[col] = pd.to_numeric(results[col], errors='coerce')
        
        # Rule-based anomalies
        results['bp_ratio'] = np.where(
            results['diastolic_bp'].notna() & (results['diastolic_bp'] != 0),
            results['systolic_bp'] / results['diastolic_bp'],
            np.nan
        )
        
        conditions = {
            'High SBP': results['systolic_bp'] > 180,
            'Low DBP': results['diastolic_bp'] < 40,
            'Invalid Ratio': (results['bp_ratio'] < 0.7) | (results['bp_ratio'] > 3.5),
            'Missing Age': results['age'].isna(),
            'Extreme Age': (results['age'] < 18) | (results['age'] > 120),
            'Gender Outlier': ~results['gender'].isin(['M', 'F', 'Male', 'Female']) 
                            if 'gender' in results.columns 
                            else pd.Series(False, index=results.index)
        }
        
        # ML Anomaly Detection
        features = results[numeric_cols].fillna(0)
        clf = IsolationForest(contamination=threshold, random_state=42)
        results['ml_anomaly'] = clf.fit_predict(features) == -1
        
        # Combine results
        for name, condition in conditions.items():
            results[name] = condition
        results['is_anomaly'] = results[list(conditions.keys())].any(axis=1) | results['ml_anomaly']
        
        return results, list(conditions.keys())
        
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return pd.DataFrame(), []

# --- Database Operations ---
def save_to_db(df):
    """Safe database insertion with automatic column alignment"""
    try:
        conn = sqlite3.connect('clinical_data.db')
        
        # Get database schema
        cursor = conn.execute("PRAGMA table_info(patients)")
        db_columns = [col[1] for col in cursor.fetchall() if col[1] not in ('id', 'timestamp')]
        
        # Prepare data - only keep columns that exist in database
        columns_to_save = [col for col in db_columns if col in df.columns]
        filtered_df = df[columns_to_save].copy()
        
        # Insert data
        filtered_df.to_sql('patients', conn, if_exists='append', index=False)
    except Exception as e:
        st.error(f"Database error: {str(e)}")
    finally:
        if conn:
            conn.close()

def generate_report(df, format='pdf'):
    """Create automated reports"""
    try:
        if format == 'pdf':
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Clinical Data Report", ln=1, align='C')
            
            # Add table headers
            pdf.cell(40, 10, "Patient ID", 1)
            pdf.cell(20, 10, "Age", 1)
            pdf.cell(30, 10, "BP (S/D)", 1)
            pdf.cell(40, 10, "Diagnosis", 1)
            pdf.cell(40, 10, "Visit Date", 1)
            pdf.ln()
            
            # Add data rows
            for _, row in df.iterrows():
                pdf.cell(40, 10, str(row['patient_id']), 1)
                pdf.cell(20, 10, str(row['age']), 1)
                pdf.cell(30, 10, f"{row['systolic_bp']}/{row['diastolic_bp']}", 1)
                pdf.cell(40, 10, str(row['diagnosis']), 1)
                pdf.cell(40, 10, str(row['visit_date']), 1)
                pdf.ln()
                
            pdf.output("report.pdf")
            return open("report.pdf", "rb")
        return df.to_csv(index=False).encode()
    except Exception as e:
        st.error(f"Report generation failed: {str(e)}")
        return None

# --- UI Components ---
def show_metrics(df, anomalies):
    """Enhanced dashboard"""
    st.subheader("🧪 Data Quality Report")
    cols = st.columns(4)
    cols[0].metric("Total Records", len(df))
    cols[1].metric("Anomalies", f"{df['is_anomaly'].sum()} ({df['is_anomaly'].mean()*100:.1f}%)")
    cols[2].metric("ML Detected", f"{df['ml_anomaly'].sum()}")
    cols[3].metric("Top Issue", df[anomalies].sum().idxmax())

# --- Main App ---
def main():
    # Initialize
    init_db()
    name, auth_status, username = setup_auth()
    
    # Authentication flow
    if not auth_status:
        st.warning("Please login to access the dashboard")
        if auth_status is False:
            st.error("Invalid credentials")
        return
    
    # Main app interface
    st.title(f"🏥 Clinical Data Dashboard | Welcome {name}")
    
    with st.sidebar:
        st.session_state.authenticator.logout("Logout", "sidebar")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        sensitivity = st.slider("Anomaly Sensitivity", 0.1, 0.5, 0.2, 0.05)
    
    if uploaded_file:
        try:
            # Process data
            raw_df = pd.read_csv(uploaded_file)
            processed_df, anomaly_types = analyze_data(raw_df, sensitivity)
            
            # Ensure visit_date exists (for backward compatibility)
            if 'visit_date' not in processed_df.columns:
                processed_df['visit_date'] = datetime.now().strftime('%Y-%m-%d')
            
            save_to_db(processed_df)
            
            # Display results
            st.success(f"✅ Analyzed {len(processed_df)} records")
            show_metrics(processed_df, anomaly_types)
            
            # Tabs interface
            tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Analysis", "📥 Export"])
            
            with tab1:
                st.dataframe(processed_df.describe())
            
            with tab2:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(
                    data=processed_df, 
                    x='age', 
                    y='systolic_bp',
                    hue='is_anomaly',
                    palette={True: 'red', False: 'green'},
                    ax=ax
                )
                ax.set_title("Blood Pressure Analysis")
                st.pyplot(fig)
            
            with tab3:
                export_format = st.radio("Format", ["PDF", "CSV"], horizontal=True)
                if st.button("Generate Report"):
                    with st.spinner("Creating report..."):
                        report = generate_report(processed_df, export_format.lower())
                        if report:
                            st.download_button(
                                f"Download {export_format}",
                                report,
                                f"clinical_report_{datetime.now().date()}.{export_format.lower()}"
                            )
        
        except Exception as e:
            st.error(f"❌ Processing error: {str(e)}")

if __name__ == "__main__":
    main()