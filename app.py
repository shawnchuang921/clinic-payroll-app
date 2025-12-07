import streamlit as st
import pandas as pd
import re
import io
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="Clinic Payroll Reconciler", layout="wide")

st.title("🏥 Clinic Staff vs. Sales Reconciler (Dual Pay Structure)")
st.markdown("""
Upload the **Staff Log (combined Hourly & Percentage)** and the **Sales Record** below. 
The summary is now split based on the staff's pay structure, showing the most relevant errors for each group.
""")

# --- Sidebar: File Uploads ---
st.sidebar.header("1. Upload Data")
staff_file = st.sidebar.file_uploader("Upload Combined Staff Log (CSV)", type=['csv'])
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

def calculate_keyword_score(row):
    """Gives a bonus score if the Staff Notes and Sales Item share key service words."""
    if pd.isna(row.get('notes')) or pd.isna(row.get('item')):
        return 0
    
    staff_note = str(row['notes']).lower()
    sales_item = str(row['item']).lower()
    
    keywords = ['report', 'assessment', 'intervention', 'session', 'consultation', 'writing']
    score = 0
    
    for kw in keywords:
        if kw in staff_note and kw in sales_item:
            score += 10
            
    return score

def extract_expected_hours(item):
    """Parses the session length (e.g., '60 mins') from the sales item description."""
    if not isinstance(item, str):
        return None
    item = item.lower()
    
    match_min = re.search(r'(\d+)\s*mins?$', item)
    if match_min:
        minutes = int(match_min.group(1))
        return minutes / 60.0 
    
    match_hr = re.search(r'(\d+)\s*hours?$', item)
    if match_hr:
        return float(match_hr.group(1))

    return None

def check_hours_validation(row):
    if row['Status'] != 'Matched':
        return 'N/A'
    
    staff_hrs = row.get('direct_hrs')
    expected_hrs = row.get('expected_hours')
    
    if pd.isna(staff_hrs) and pd.isna(expected_hrs): return 'Missing Data'
    if pd.isna(staff_hrs) or pd.isna(expected_hrs): return 'Missing Staff/Sales Hrs'
    
    if round(staff_hrs, 2) == round(expected_hrs, 2):
        return 'Match'
    else:
        return f'Mismatch: Staff Hrs ({staff_hrs}) != Expected Hrs ({expected_hrs})'

def check_amount_final(row):
    if row['Status'] == 'Matched':
        if row.get('charged_amount') == row.get('subtotal'):
            return 'Match'
        elif row.get('total_pay') == row.get('subtotal'):
            return 'Mismatch (Pay Match)'
        else:
            return 'Mismatch'
    return 'N/A'
    
def get_staff_pay_types(df_staff):
    """
    Categorizes staff as 'Hourly' or 'Percentage' based on their dominant pay calculation column.
    
    Hourly: Direct_Pay is the dominant paid amount (used for Direct_Hrs * Direct_Pay).
    Percentage: Percentage_Pay is the dominant paid amount.
    """
    # Ensure numeric types, coercing errors
    numeric_cols = ['direct_pay', 'percentage_pay']
    for col in numeric_cols:
        df_staff[col] = pd.to_numeric(df_staff[col], errors='coerce').fillna(0)
    
    staff_pay_totals = df_staff.groupby('staff_name')[numeric_cols].sum()
    
    def determine_pay_type(row):
        # A staff member is Hourly if the sum of their Direct_Pay is greater than or equal
        # to the sum of their Percentage_Pay. This relies on the convention that one
        # column is zeroed out for the non-applicable pay type.
        if row['direct_pay'] >= row['percentage_pay']:
            return 'Hourly'
        else:
            return 'Percentage'

    staff_pay_types = staff_pay_totals.apply(determine_pay_type, axis=1).reset_index(name='Pay_Type')
    return staff_pay_types[['staff_name', 'Pay_Type']]


# --- Run App ---
if staff_file and sales_file:
    st.sidebar.success("Files uploaded successfully!")
    
    if st.button("Run Reconciliation"):
        try:
            with st.spinner('Processing records with Smart Matching...'):
                # 1. Load Data
                df_staff = pd.read_csv(staff_file, encoding='latin1', engine='python', on_bad_lines='skip')
                df_sales = pd.read_csv(sales_file, encoding='latin1', engine='python', on_bad_lines='skip')

                # --- PREPROCESSING & CLEANING ---
                
                # Standardize Columns
                df_staff.columns = df_staff.columns.str.strip().str.replace(' ', '_').str.lower()
                df_sales.columns = df_sales.columns.str.strip().str.replace(' ', '_').str.lower()
                
                # Positional Fix for Staff Date
                if len(df_staff.columns) > 0:
                    df_staff = df_staff.rename(columns={df_staff.columns[0]: 'date'}, errors='ignore')

                # NEW: Determine Staff Pay Type and Merge
                df_pay_types = get_staff_pay_types(df_staff.copy())
                df_staff = pd.merge(df_staff, df_pay_types, on='staff_name', how='left')

                # Create Unique IDs (Crucial for 1-to-1 matching)
                df_staff['staff_id'] = df_staff.index
                df_sales['sales_id'] = df_sales.index

                # Process Sales Dates & Names
                df_sales = df_sales.dropna(subset=['patient', 'invoice_date'])
                df_sales['dt_obj'] = pd.to_datetime(df_sales['invoice_date'], utc=True)
                df_sales['dt_local'] = df_sales['dt_obj'].dt.tz_convert('America/Vancouver')
                df_sales['date_str'] = df_sales['dt_local'].dt.normalize().astype(str).str[:10]
                df_sales['patient_norm'] = df_sales['patient'].apply(clean_name_string) 
                df_sales['expected_hours'] = df_sales['item'].apply(extract_expected_hours)

                # Process Staff Dates & Names
                df_staff['date_obj'] = pd.to_datetime(df_staff['date'])
                df_staff['date_str'] = df_staff['date_obj'].dt.normalize().astype(str).str[:10]
                df_staff['extracted_name'] = df_staff['notes'].apply(extract_name)
                df_staff['name_norm'] = df_staff['extracted_name'].apply(clean_name_string)

                # --- MATCHING LOGIC (Same as before) ---
                potential_matches = pd.merge(
                    df_staff, df_sales, left_on=['name_norm'], right_on=['patient_norm'], 
                    how='outer', suffixes=('_staff', '_sales')
                )
                
                def get_date_diff(row):
                    if pd.notna(row['date_obj']) and pd.notna(row['dt_obj']):
                        return abs((row['date_obj'].date() - row['dt_obj'].date()).days)
                    return 999 

                potential_matches['date_diff'] = potential_matches.apply(get_date_diff, axis=1)
                candidates = potential_matches[potential_matches['date_diff'] <= 1].copy()
                candidates['service_score'] = candidates.apply(calculate_keyword_score, axis=1)
                candidates = candidates.sort_values(by=['service_score', 'date_diff'], ascending=[False, True])
                
                matched_staff_ids = set()
                matched_sales_ids = set()
                final_rows = []

                for _, row in candidates.iterrows():
                    sid, slid = row['staff_id'], row['sales_id']
                    if sid not in matched_staff_ids and slid not in matched_sales_ids:
                        match_dict = row.to_dict()
                        match_dict['Status'] = 'Matched'
                        match_dict['Match_Type'] = f"Match (Diff: {row['date_diff']} days)"
                        final_rows.append(match_dict)
                        matched_staff_ids.add(sid)
                        matched_sales_ids.add(slid)

                # Handle Unmatched Records (Keep Pay_Type consistent)
                unmatched_staff = df_staff[~df_staff['staff_id'].isin(matched_staff_ids)].copy()
                for _, row in unmatched_staff.iterrows():
                    new_row = row.to_dict()
                    if 'date_str' in new_row: new_row['date_str_staff'] = new_row.pop('date_str')
                    new_row['Status'], new_row['Match_Type'] = 'In Staff Log Only (Missing in Sales)', 'N/A'
                    new_row['invoice_date'], new_row['patient'], new_row['item'], new_row['subtotal'], new_row['expected_hours'] = None, None, None, None, None
                    final_rows.append(new_row)

                unmatched_sales = df_sales[~df_sales['sales_id'].isin(matched_sales_ids)].copy()
                for _, row in unmatched_sales.iterrows():
                    new_row = row.to_dict()
                    new_row['date_str_sales'] = new_row.pop('date_str')
                    new_row['date_str_staff'], new_row['date'], new_row['extracted_name'], new_row['notes'], new_row['charged_amount'], new_row['direct_hrs'] = None, None, None, None, None, None
                    new_row['Pay_Type'] = None # Cannot determine pay type for unmatched sales
                    new_row['Status'], new_row['Match_Type'] = 'In Sales Record Only (Missing in Log)', 'N/A'
                    final_rows.append(new_row)

                final_df = pd.DataFrame(final_rows)
                
                # --- FINAL REPORT FORMATTING ---
                final_df['Amount_Status'] = final_df.apply(check_amount_final, axis=1)
                final_df['Hours_Validation_Status'] = final_df.apply(check_hours_validation, axis=1)
                final_df['Display_Date'] = final_df['date_str_staff'].fillna(final_df['date_str_sales'])

                report_cols = ['Display_Date', 'extracted_name', 'Pay_Type', 'notes', 'direct_hrs', 'charged_amount', 
                               'invoice_date', 'patient', 'item', 'expected_hours', 'subtotal', 
                               'Status', 'Amount_Status', 'Hours_Validation_Status', 'Match_Type']
                
                rename_map = {
                    'Display_Date': 'Date (Staff Log)', 
                    'extracted_name': 'Staff_Patient_Name',
                    'Pay_Type': 'Staff_Pay_Type', # NEW
                    'notes': 'Notes', 
                    'direct_hrs': 'Staff_Direct_Hrs', 
                    'charged_amount': 'Charged_Amount',
                    'invoice_date': 'Invoice Date', 
                    'patient': 'Patient',
                    'item': 'Item', 
                    'expected_hours': 'Sales_Expected_Hrs', 
                    'subtotal': 'Subtotal',
                    'Match_Type': 'Date_Tolerance'
                }

                available_cols = [c for c in report_cols if c in final_df.columns]
                final_report = final_df[available_cols].rename(columns=rename_map)

                # --- METRICS CALCULATION ---
                
                # Split the final report by Pay_Type
                df_hourly = final_report[final_report['Staff_Pay_Type'] == 'Hourly'].copy()
                df_percentage = final_report[final_report['Staff_Pay_Type'] == 'Percentage'].copy()
                
                def calculate_metrics(df, error_type):
                    matched_count = len(df[df['Status'] == 'Matched'])
                    staff_only_count = len(df[df['Status'] == 'In Staff Log Only (Missing in Sales)'])
                    sales_only_count = len(df[df['Status'] == 'In Sales Record Only (Missing in Log)'])
                    
                    if error_type == 'Hours':
                        error_count = len(df[
                            (df['Status'] == 'Matched') & 
                            (df['Hours_Validation_Status'].str.startswith('Mismatch', na=False))
                        ])
                    elif error_type == 'Amount':
                        error_count = len(df[
                            (df['Status'] == 'Matched') & 
                            (df['Amount_Status'].str.startswith('Mismatch', na=False))
                        ])
                    else:
                        error_count = 0
                        
                    return matched_count, error_count, staff_only_count, sales_only_count
                
                h_matched, h_error, h_staff_only, h_sales_only = calculate_metrics(df_hourly, 'Hours')
                p_matched, p_error, p_staff_only, p_sales_only = calculate_metrics(df_percentage, 'Amount')

                # --- Display Results ---
                
                st.markdown("### 📊 Reconciliation Summary by Pay Structure")
                
                # Hourly Staff Summary
                st.markdown("#### Hourly Rate Staff Summary")
                col1, col2, col3, col4 = st.columns(4) 
                col1.metric("Total Matches (1-to-1)", h_matched)
                col2.metric("**Hours Mismatch Errors**", h_error) 
                col3.metric("In Staff Log Only", h_staff_only)
                col4.metric("In Sales Record Only", h_sales_only)

                st.markdown("---")
                
                # Percentage Pay Staff Summary
                st.markdown("#### Percentage Pay Staff Summary")
                col1, col2, col3, col4 = st.columns(4) 
                col1.metric("Total Matches (1-to-1)", p_matched)
                col2.metric("**Amount Mismatch Errors**", p_error) 
                col3.metric("In Staff Log Only", p_staff_only)
                col4.metric("In Sales Record Only", p_sales_only)
                
                st.markdown("---")

                st.markdown("### 📋 Detailed Report")
                st.dataframe(final_report, use_container_width=True)

                csv = final_report.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Report as CSV",
                    data=csv,
                    file_name='Reconciliation_Report_Dual_Pay.csv',
                    mime='text/csv',
                )

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.warning("A critical error prevents processing. Please ensure your combined Staff Log CSV file is correctly formatted.")

else:
    st.info("👋 Please upload both CSV files in the sidebar to begin. Remember to combine all staff logs into a single file for the new dual-pay logic.")
