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

# --- Processing Logic (Functions remain the same) ---
def extract_name(note):
    """Clean name from notes by removing time patterns."""
    if not isinstance(note, str): return ""
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
                # 1. Load Data with resilience fixes
                df_staff = pd.read_csv(staff_file, encoding='latin1', engine='python', on_bad_lines='skip')
                df_sales = pd.read_csv(sales_file, encoding='latin1', engine='python', on_bad_lines='skip')

                # --- NEW RESILIENCE BLOCK: Clean Column Names ---
                # This fixes invisible spaces or case mismatches in headers
                df_staff.columns = df_staff.columns.str.strip().str.replace(' ', '_').str.lower()
                df_sales.columns = df_sales.columns.str.strip().str.replace(' ', '_').str.lower()
                
                # We update the expected column names to reflect the cleaned, lowercase, underscore format.
                
                # 2. Preprocessing Sales
                # NOTE: Columns are now lowercase with underscores!
                df_sales = df_sales.dropna(subset=['patient', 'invoice_date'])
                df_sales['dt_obj'] = pd.to_datetime(df_sales['invoice_date'], utc=True)
                # Convert to Vancouver time
                df_sales['dt_local'] = df_sales['dt_obj'].dt.tz_convert('America/Vancouver')
                df_sales['date_str'] = df_sales['dt_local'].dt.date.astype(str)
                df_sales['patient_norm'] = df_sales['patient'].astype(str).str.strip().str.lower()

                # 3. Preprocessing Staff
                # NOTE: Columns are now lowercase with underscores!
                df_staff['date_obj'] = pd.to_datetime(df_staff['date'])
                df_staff['date_str'] = df_staff['date_obj'].dt.date.astype(str)
                df_staff['extracted_name'] = df_staff['notes'].apply(extract_name)
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

                # 5. Labeling and Amount Check (Uses lowercase column names)
                status_map = {
                    'both': 'Matched',
                    'left_only': 'In Staff Log Only (Missing in Sales)',
                    'right_only': 'In Sales Record Only (Missing in Log)'
                }
                merged_df['Status'] = merged_df['_merge'].map(status_map)
                
                # Update check_amount to use lowercase names
                def check_amount_v2(row):
                    if row['Status'] == 'Matched':
                        # Check charged_amount vs subtotal
                        if row.get('charged_amount') == row.get('subtotal'):
                            return 'Match'
                        else:
                            return 'Mismatch'
                    return 'N/A'

                merged_df['Amount_Status'] = merged_df.apply(check_amount_v2, axis=1)

                # 6. Final Report Columns (Original capitalization for display)
                desired_cols = ['date', 'extracted_name', 'notes', 'charged_amount',
                                'invoice_date', 'patient', 'item', 'subtotal', 'Status', 'Amount_Status']
                
                rename_map = {'date': 'Date', 'extracted_name': 'Staff_Patient_Name',
                              'notes': 'Notes', 'charged_amount': 'Charged_Amount',
                              'invoice_date': 'Invoice Date', 'patient': 'Patient',
                              'item': 'Item', 'subtotal': 'Subtotal'}
                
                # Use the lowercase 'date' column for merging/filling
                if 'date' in merged_df.columns:
                     merged_df['date'] = merged_df['date'].fillna(merged_df['date_str'])

                final_report = merged_df[desired_cols].rename(columns=rename_map)

                # --- Display Results ---
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
            # Display a more user-friendly error message
            st.error(f"An error occurred: {e}")
            st.warning("Please ensure your CSV files have the correct column headers: Date, Notes, Charged_Amount, Patient, Invoice Date, Item, and Subtotal.")

else:
    st.info("👋 Please upload both CSV files in the sidebar to begin.")
