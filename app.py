import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time

# Set page config
st.set_page_config(
    page_title="Clinical Data AI Cleaner",
    page_icon="🏥",
    layout="wide"
)

# Title and description
st.title("🏥 Clinical Data Quality Dashboard")
st.markdown("""
An AI-powered tool for detecting anomalies in clinical trial data.
Upload your dataset below to analyze data quality.
""")

# Sidebar for upload and settings
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Clinical trial data in CSV format"
    )
    
    st.header("Settings")
    anomaly_threshold = st.slider(
        "Anomaly Detection Sensitivity",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
        help="Higher values catch more potential errors (but may increase false positives)"
    )
    
    st.markdown("---")
    st.markdown("**Made with ❤️ for Clinical Data Management**")

# Main analysis function (simplified version from notebook)
def analyze_data(df):
    """Process data and detect anomalies"""
    results = df.copy()
    
    # Convert numeric columns
    numeric_cols = ['age', 'systolic_bp', 'diastolic_bp']
    for col in numeric_cols:
        if col in results.columns:
            results[col] = pd.to_numeric(results[col], errors='coerce')
    
    # Basic anomaly detection (simplified)
    results['bp_ratio'] = results['systolic_bp'] / results['diastolic_bp']
    
    # Flag numeric anomalies
    conditions = {
        'High SBP': results['systolic_bp'] > 180,
        'Low DBP': results['diastolic_bp'] < 40,
        'Invalid Ratio': results['bp_ratio'] < 0.7,
        'Missing Age': results['age'].isna()
    }
    
    for name, condition in conditions.items():
        results[name] = condition
    
    # Calculate overall anomaly score
    results['anomaly_score'] = results[list(conditions.keys())].mean(axis=1)
    results['is_anomaly'] = results['anomaly_score'] > anomaly_threshold
    
    return results

# Main app logic
if uploaded_file is not None:
    try:
        # Load and process data
        with st.spinner("Analyzing data..."):
            raw_df = pd.read_csv(uploaded_file)
            processed_df = analyze_data(raw_df)
            time.sleep(1)  # Simulate processing time
            
        # Show success message
        st.success("Analysis complete!")
        
        # Metrics row
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", len(processed_df))
        col2.metric("Records with Issues", 
                   f"{processed_df['is_anomaly'].sum()}",
                   f"{processed_df['is_anomaly'].mean()*100:.1f}%")
        col3.metric("Most Common Issue", 
                   processed_df[list(conditions.keys())].sum().idxmax())
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Detailed Analysis", "📥 Export"])
        
        with tab1:
            # Data summary
            st.subheader("Data Summary")
            st.dataframe(processed_df.describe(), use_container_width=True)
            
            # Anomaly distribution
            st.subheader("Anomaly Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x='is_anomaly', data=processed_df, ax=ax)
            ax.set_xticklabels(['Clean', 'Anomaly'])
            st.pyplot(fig)
        
        with tab2:
            # Interactive filtering
            st.subheader("Filter Anomalies")
            issue_type = st.selectbox(
                "Select issue type to inspect",
                options=list(conditions.keys()) + ['All']
            )
            
            if issue_type != 'All':
                filtered_df = processed_df[processed_df[issue_type]]
            else:
                filtered_df = processed_df[processed_df['is_anomaly']]
            
            st.dataframe(filtered_df, use_container_width=True)
            
            # Blood pressure scatter plot
            st.subheader("Blood Pressure Analysis")
            fig, ax = plt.subplots()
            sns.scatterplot(
                x='systolic_bp',
                y='diastolic_bp',
                hue='is_anomaly',
                data=processed_df,
                palette={True: 'red', False: 'green'},
                ax=ax
            )
            ax.axhline(40, color='orange', linestyle='--')
            ax.axvline(180, color='orange', linestyle='--')
            st.pyplot(fig)
        
        with tab3:
            # Export options
            st.subheader("Export Cleaned Data")
            
            export_format = st.radio(
                "Select export format",
                options=["CSV", "Excel"]
            )
            
            if st.button("Generate Export"):
                with st.spinner("Preparing download..."):
                    if export_format == "CSV":
                        csv = processed_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"cleaned_data_{datetime.now().date()}.csv",
                            mime="text/csv"
                        )
                    else:
                        excel = processed_df.to_excel(index=False)
                        st.download_button(
                            label="Download Excel",
                            data=excel,
                            file_name=f"cleaned_data_{datetime.now().date()}.xlsx",
                            mime="application/vnd.ms-excel"
                        )
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    # Demo mode with sample data
    if st.checkbox("Use sample data for demo"):
        with st.spinner("Loading sample data..."):
            # Generate synthetic data
            def generate_sample_data():
                data = {
                    'patient_id': [f"PT-{1000+i}" for i in range(50)],
                    'age': np.random.randint(18, 90, 50),
                    'gender': np.random.choice(['M', 'F'], 50),
                    'systolic_bp': np.random.normal(120, 25, 50),
                    'diastolic_bp': np.random.normal(80, 15, 50),
                    'diagnosis': np.random.choice(['Diabetes', 'HTN', 'Asthma', 'Cancer'], 50)
                }
                df = pd.DataFrame(data)
                
                # Introduce some anomalies
                df.loc[3:5, 'systolic_bp'] = 220
                df.loc[10:12, 'diastolic_bp'] = 30
                df.loc[15, 'age'] = 150
                return df
            
            sample_df = generate_sample_data()
            processed_df = analyze_data(sample_df)
            
            # Show sample data preview
            st.subheader("Sample Data Preview")
            st.dataframe(sample_df.head())
            
            st.info("This is demo data. Upload your own CSV for real analysis.")
