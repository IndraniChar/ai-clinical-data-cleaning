import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def main():
    st.title("Clinical Data Dashboard")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        try:
            # Read and display data
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
            
            # Basic analysis
            st.subheader("Basic Statistics")
            st.write(df.describe())
            
            # Visualization
            st.subheader("Data Visualization")
            if 'age' in df.columns and 'systolic_bp' in df.columns:
                fig, ax = plt.subplots()
                ax.scatter(df['age'], df['systolic_bp'])
                ax.set_xlabel('Age')
                ax.set_ylabel('Systolic BP')
                st.pyplot(fig)
            else:
                st.warning("No age/systolic_bp columns for visualization")
            
            # Export options
            st.subheader("Export Data")
            if st.button("Download CSV"):
                st.download_button(
                    label="Download",
                    data=df.to_csv(index=False),
                    file_name="clinical_data.csv"
                )
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()