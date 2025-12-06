import streamlit as st
import pandas as pd
import re
import io

# --- Page Configuration ---
st.set_page_config(page_title="Clinic Payroll Reconciler", layout="wide")

st.title("🏥 Clinic Staff vs. Sales Reconciler")
st.markdown("""
Upload the **Staff Log** and the **Sales Record** below. 
The app will match them based on **Patient Name only** to account for date/timezone shifts.
""")

# --- Sidebar: File Uploads ---
st.sidebar.header("1. Upload Data")
staff_file = st.sidebar.file_uploader("Upload Staff Log (CSV)", type=['csv'])
sales_file = st.sidebar.file_uploader("Upload Sales Record (CSV)", type=['csv'])

# --- Helper Functions ---
def clean_name_string(name):
    """Aggressively removes all non-alphabetic characters for reliable fuzzy matching."""
    if not isinstance(name, str): return ""
    cleaned = re.sub(r'[^a-z]', '', name.lower())
    return cleaned

def extract_name(note):
    """Clean name from notes by removing time patterns."""
    if not isinstance(note, str): return ""
    pattern = r'\s+\d{1,2}(?::\d{2})?-\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)?.*$'
    clean_name = re.sub(pattern, '', note, flags=re.IGNORECASE)
    return clean_name.strip()

def check_amount_v2(row):
    """Compare charged amount vs subtotal using lowercase names."""
    if row['Status'] == 'Matched':
        if row.get('charged_amount') == row.get('subtotal'):
            return 'Match'
        else:
            return 'Mismatch'
    return 'N/A'

def check_date_tolerance(row, tolerance_days=1):
    """Checks if the Staff Date and Sales Invoice Date are within the tolerance."""
    if row['Status'] == 'Matched':
        staff_date = row.get('date_obj')
        sales_date = row.get('dt_obj')
        
        if pd.notna(staff_date) and pd.notna(sales_date):
            diff = abs((staff_date.date() - sales_date.date()).days)
            if diff <= tolerance_days:
                return f"Match (Diff: {diff} days)"
            else:
                return f"Date Mismatch (Diff: {diff} days)"
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

                # --- RESILIENCE BLOCK: Clean Column Names ---
                df_staff.columns = df_staff.columns.str.strip().str.replace(' ', '_').str.lower()
                df_sales.columns = df_sales.columns.str.strip().str.replace(' ', '_').str.lower()
                
                # Positional Fix
                if len(df_staff.columns) > 0:
                    df_staff = df_staff.rename(columns={df_staff.columns[0]: 'date'}, errors='ignore')

                # 2. Preprocessing Sales
                df_sales = df_sales.dropna(subset=['patient', 'invoice_date'])
                df_sales['dt_obj'] = pd.to_datetime(df_sales['invoice_date'], utc=True)
                df_sales['dt_local'] = df_sales['dt_obj'].dt.tz_convert('America/Vancouver')
                df_sales['date_str'] = df_sales['dt_local'].dt.normalize().astype(str).str[:10]
                df_sales['patient_norm'] = df_sales['patient'].apply(clean_name_string) 

                # 3. Preprocessing Staff
                df_staff['date_obj'] = pd.to_datetime(df_staff['date'])
                df_staff['date_str'] = df_staff['date_obj'].dt.normalize().astype(str).str[:10]
                df_staff['extracted_name'] = df_staff['notes'].apply(extract_name)
                df_staff['name_norm'] = df_staff['extracted_name'].apply(clean_name_string)

                # 4. Merging - NAME ONLY
                merged_df = pd.merge(
                    df_staff,
                    df_sales,
                    left_on=['name_norm'],
                    right_on=['patient_norm'],
                    how='outer',
                    indicator=True
                )

                # 5. Labeling, Amount, and Date Tolerance Check
                status_map = {
                    'both': 'Matched',
                    'left_only': 'In Staff Log Only (Missing in Sales)',
                    'right_only': 'In Sales Record Only (Missing in Log)'
                }
                merged_df['Status'] = merged_df['_merge'].map(status_map)
                merged_df['Amount_Status'] = merged_df.apply(check_amount_v2, axis=1)
                merged_df['Date_Tolerance_Status'] = merged_df.apply(check_date_tolerance, axis=1)

                # 6. Final Report Columns
                merged_df['Display_Date'] = merged_df['date_str_x'].fillna(merged_df['date_str_y'])
                
                report_cols = ['Display_Date', 'extracted_name', 'notes', 'charged_amount',
                               'invoice_date', 'patient', 'item', 'subtotal', 
                               'Status', 'Amount_Status', 'Date_Tolerance_Status'] 

                final_report = merged_df[report_cols].rename(columns={
                    'Display_Date': 'Date (Staff Log)', 
                    'extracted_name': 'Staff_Patient_Name',
                    'notes': 'Notes', 
                    'charged_amount': 'Charged_Amount',
                    'invoice_date': 'Invoice Date', 
                    'patient': 'Patient',
                    'item': 'Item', 
                    'subtotal': 'Subtotal'
                })

                # --- NEW ROBUST METRIC CALCULATIONS ---
                
                # Metric 1: Matches (within 1 day)
                matched_count = len(final_report[
                    (final_report['Status'] == 'Matched') & 
                    (final_report['Date_Tolerance_Status'].str.contains("Match", na=False))
                ])

                # Metric 2: Staff Log Only (Exact match on the status string)
                staff_only_count = len(final_report[
                    final_report['Status'] == 'In Staff Log Only (Missing in Sales)'
                ])

                # Metric 3: Sales Record Only (Exact match on the status string)
                sales_only_count = len(final_report[
                    final_report['Status'] == 'In Sales Record Only (Missing in Log)'
                ])

                # --- Display Results and Download Button ---
                st.markdown("### 📊 Reconciliation Summary")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Matches (Within 1 Day)", matched_count)
                col2.metric("In Staff Log Only", staff_only_count)
                col3.metric("In Sales Record Only", sales_only_count) # NOW CORRECT

                st.markdown("### 📋 Detailed Report")
                st.dataframe(final_report, use_container_width=True)

                csv = final_report.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Report as CSV",
                    data=csv,
                    file_name='Reconciliation_Report.csv',
                    mime='text/csv',
                )

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.warning("A critical error prevents processing. Please ensure your CSV files have the correct header names and file contents.")

else:
    st.info("👋 Please upload both CSV files in the sidebar to begin.")
