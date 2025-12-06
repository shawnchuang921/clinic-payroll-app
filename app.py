import streamlit as st
import pandas as pd
import re
import io

# --- Page Configuration ---
st.set_page_config(page_title="Clinic Payroll Reconciler", layout="wide")

st.title("🏥 Clinic Staff vs. Sales Reconciler")
st.markdown("""
Upload the **Staff Log** and the **Sales Record** below. 
The app will match them based on Date and Patient Name and generate a report.
""")

# --- Sidebar: File Uploads ---
st.sidebar.header("1. Upload Data")
staff_file = st.sidebar.file_uploader("Upload Staff Log (CSV)", type=['csv'])
sales_file = st.sidebar.file_uploader("Upload Sales Record (CSV)", type=['csv'])

# --- Processing Logic ---
def extract_name(note):
    """Clean name from notes by removing time patterns."""
    if not isinstance(note, str): return ""
    # Regex to remove time patterns like "1-3 pm", "11:30-12 pm"
    pattern = r'\s+\d{1,2}(?::\d{2})?-\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)?.*$'
    clean_name = re.sub(pattern, '', note, flags=re.IGNORECASE)
    return clean_name.strip()

def check_amount(row):
    """Compare charged amount vs subtotal."""
    if row['Status'] == 'Matched':
        if row.get('Charged_Amount') == row.get('Subtotal'):
            return 'Match'
        else:
            return 'Mismatch'
    return 'N/A'

# --- Run App ---
if staff_file and sales_file:
    st.sidebar.success("Files uploaded successfully!")
    
    if st.button("Run Reconciliation"):
        try:
            with st.spinner('Processing records...'):
                # 1. Load Data
                df_staff = pd.read_csv(staff_file)
                df_sales = pd.read_csv(sales_file)

                # 2. Preprocessing Sales
                df_sales = df_sales.dropna(subset=['Patient', 'Invoice Date'])
                df_sales['dt_obj'] = pd.to_datetime(df_sales['Invoice Date'], utc=True)
                # Convert to Vancouver time
                df_sales['dt_local'] = df_sales['dt_obj'].dt.tz_convert('America/Vancouver')
                df_sales['date_str'] = df_sales['dt_local'].dt.date.astype(str)
                df_sales['patient_norm'] = df_sales['Patient'].astype(str).str.strip().str.lower()

                # 3. Preprocessing Staff
                df_staff['date_obj'] = pd.to_datetime(df_staff['Date'])
                df_staff['date_str'] = df_staff['date_obj'].dt.date.astype(str)
                df_staff['extracted_name'] = df_staff['Notes'].apply(extract_name)
                df_staff['name_norm'] = df_staff['extracted_name'].str.lower()

                # 4. Merging
                merged_df = pd.merge(
                    df_staff,
                    df_sales,
                    left_on=['date_str', 'name_norm'],
                    right_on=['date_str', 'patient_norm'],
                    how='outer',
                    indicator=True
                )

                # 5. Labeling
                status_map = {
                    'both': 'Matched',
                    'left_only': 'In Staff Log Only (Missing in Sales)',
                    'right_only': 'In Sales Record Only (Missing in Log)'
                }
                merged_df['Status'] = merged_df['_merge'].map(status_map)
                merged_df['Amount_Status'] = merged_df.apply(check_amount, axis=1)

                # 6. Final Report Columns
                # Ensure columns exist before selecting them to avoid errors
                available_cols = merged_df.columns.tolist()
                
                # Define ideal column order, but filter based on what actually exists
                desired_cols = ['Date_x', 'extracted_name', 'Notes', 'Charged_Amount',
                                'Invoice Date', 'Patient', 'Item', 'Subtotal', 'Status', 'Amount_Status']
                
                # Rename dict
                rename_map = {'Date_x': 'Date', 'extracted_name': 'Staff_Patient_Name'}
                
                # Handle cases where Date_x might be NaN (from right_only merge), fill with sales date
                if 'Date_x' in merged_df.columns and 'date_str' in merged_df.columns:
                     merged_df['Date_x'] = merged_df['Date_x'].fillna(merged_df['date_str'])

                final_report = merged_df[desired_cols].rename(columns=rename_map)

                # --- Display Results ---
                
                # Metrics
                st.markdown("### 📊 Reconciliation Summary")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Matches", len(final_report[final_report['Status']=='Matched']))
                col2.metric("Missing in Sales", len(final_report[final_report['Status'].str.contains('In Staff Log')]))
                col3.metric("Missing in Log", len(final_report[final_report['Status'].str.contains('In Sales')]))

                st.markdown("### 📋 Detailed Report")
                st.dataframe(final_report, use_container_width=True)

                # --- Download Button ---
                csv = final_report.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Report as CSV",
                    data=csv,
                    file_name='Reconciliation_Report.csv',
                    mime='text/csv',
                )

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.warning("Please check that your CSV files have the correct column names (Date, Patient, Invoice Date, etc.)")

else:
    st.info("👋 Please upload both CSV files in the sidebar to begin.")