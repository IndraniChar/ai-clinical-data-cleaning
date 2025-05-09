import streamlit as st
import pandas as pd

def main():
    st.title("Clinical Data Dashboard")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
            st.write(df)
            
            if st.button("Show Basic Stats"):
                st.write(df.describe())
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()