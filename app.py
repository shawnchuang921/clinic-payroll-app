import streamlit as st
import pandas as pd
import re
import io
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="Clinic Payroll Reconciler", layout="wide")

st.title("🏥 Clinic Staff vs. Sales Reconciler (Error Refinement)")
st.markdown("""
The reconciliation now flags **Travel Fee Discrepancies** for all staff by dynamically inferring the expected session price, correcting the issue where `Charged_Amount` might be zero in the staff log.
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
    """
    Validates Charged_Amount vs Subtotal, accounting for Travel Fee, with a fix 
    for inconsistent Charged_Amount entry (e.g., when it is 0).
    """
    if row['Status'] != 'Matched':
        return 'N/A'
    
    # Safely retrieve and default numeric values
    staff_charge = row.get('charged_amount', 0)
    travel_fee = row.get('travel_fee_used', 0)
    sales_subtotal = row.get('subtotal', 0)
    total_pay = row.get('total_pay', 0) 
    expected_hours = row.get('expected_hours', 0)
    
    # Safely retrieve and standardize Outside_Clinic status
    outside_clinic = str(row.get('outside_clinic', 'no')).strip().lower()
    
    # Calculate the expected total charge from the staff log
    staff_total_charge = staff_charge + travel_fee
    
    # --- Dynamic Base Price Assumption (used when Charged_Amount is unreliable/0) ---
    # Assume base rate is $160 per hour if expected_hours is available
    expected_base_price = expected_hours * 160.0
    
    # Check for staff claiming a home session
    staff_flags_home_session = (outside_clinic == 'yes') and (travel_fee > 0)

    # 1. Primary Check: Does Staff Total Charge (Base + Travel) match Sales Subtotal?
    if round(staff_total_charge, 2) == round(sales_subtotal, 2):
        if staff_flags_home_session:
             return 'Match (Inc. Travel Fee)'
        return 'Match'
    
    # 2. Travel Fee Mismatch Scenarios 
    
    # Scenario A: Staff log indicates a fee, but Sales Subtotal is missing it (i.e., Sales Subtotal matches base price).
    if staff_flags_home_session:
        
        # FIX FOR CHARGED_AMOUNT=0: If Charged_Amount is 0 OR the Sales Subtotal matches the dynamic base price, flag the error.
        if (round(staff_charge, 2) == 0 and round(sales_subtotal, 2) == round(expected_base_price, 2)) or \
           (round(staff_charge, 2) > 0 and round(staff_charge, 2) == round(sales_subtotal, 2)):
                
            return f'Mismatch: Staff Log indicates Home Session (+$20), but Sales Subtotal (${sales_subtotal}) is missing Travel Fee.'

    # Scenario B: Sales Subtotal suggests a fee, but Staff Log does not reflect it.
    if not staff_flags_home_session and travel_fee == 0:
        # Check if Sales Subtotal matches expected base price + 20
        if round(sales_subtotal, 2) == round(expected_base_price + 20.0, 2):
             return f'Mismatch: Sales Subtotal (${sales_subtotal}) suggests Home Session (+$20) not reflected in Staff Log.'


    # 3. Secondary Check: Does Total Pay match Sales Subtotal?
    if round(total_pay, 2) == round(sales_subtotal, 2):
        return 'Mismatch (Pay Match)' 

    # 4. Final Fallback: General Mismatch
    return f'Mismatch: Staff Total Charge (${staff_total_charge}) != Sales Subtotal (${sales_subtotal})'

def get_staff_pay_types(df_staff):
    """
    Categorizes staff as 'Hourly' or 'Percentage' based on their dominant pay calculation column.
    """
    numeric_cols = ['direct_pay', 'percentage_pay']
    for col in numeric_cols:
        df_staff[col] = pd.to_numeric(df_staff[col], errors='coerce').fillna(0)
    
    staff_pay_totals = df_staff.groupby('staff_name')[numeric_cols].sum()
    
    def determine_pay_type(row):
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
            with st.spinner('Processing records with Smart Matching and Corrected Travel Fee Logic...'):
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

                # Clean Travel Fee/Outside Clinic Columns
                df_staff['travel_fee_used'] = pd.to_numeric(df_staff['travel_fee_used'], errors='coerce').fillna(0)
                df_staff['outside_clinic'] = df_staff['outside_clinic'].astype(str).str.strip().str.lower()
                df_staff['charged_amount'] = pd.to_numeric(df_staff['charged_amount'], errors='coerce').fillna(0) # Ensure charged_amount is numeric

                # Determine Staff Pay Type and Merge
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

                # Handle Unmatched Records 
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
                    new_row['Pay_Type'], new_row['outside_clinic'], new_row['travel_fee_used'] = None, None, None 
                    new_row['Status'], new_row['Match_Type'] = 'In Sales Record Only (Missing in Log)', 'N/A'
                    final_rows.append(new_row)

                final_df = pd.DataFrame(final_rows)
                
                # --- FINAL REPORT VALIDATION & FORMATTING ---
                final_df['Amount_Status'] = final_df.apply(check_amount_final, axis=1)
                final_df['Hours_Validation_Status'] = final_df.apply(check_hours_validation, axis=1)
                final_df['Display_Date'] = final_df['date_str_staff'].fillna(final_df['date_str_sales'])

                report_cols = ['Display_Date', 'extracted_name', 'Pay_Type', 'notes', 
                               'outside_clinic', 'travel_fee_used', 'direct_hrs', 'charged_amount', 
                               'invoice_date', 'patient', 'item', 'expected_hours', 'subtotal', 
                               'Status', 'Amount_Status', 'Hours_Validation_Status', 'Match_Type']
                
                rename_map = {
                    'Display_Date': 'Date (Staff Log)', 
                    'extracted_name': 'Staff_Patient_Name',
                    'Pay_Type': 'Staff_Pay_Type',
                    'notes': 'Notes', 
                    'outside_clinic': 'Staff_Outside_Clinic',
                    'travel_fee_used': 'Staff_Travel_Fee',
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

                # --- METRICS CALCULATION (Updated for Hourly Critical Errors) ---
                
                df_hourly = final_report[final_report['Staff_Pay_Type'] == 'Hourly'].copy()
                df_percentage = final_report[final_report['Staff_Pay_Type'] == 'Percentage'].copy()
                
                def calculate_metrics(df, pay_type):
                    matched_count = len(df[df['Status'] == 'Matched'])
                    staff_only_count = len(df[df['Status'] == 'In Staff Log Only (Missing in Sales)'])
                    sales_only_count = len(df[df['Status'] == 'In Sales Record Only (Missing in Log)'])
                    
                    if pay_type == 'Hourly':
                        # 1. Hours Mismatch Errors
                        hours_mismatch = len(df[
                            (df['Status'] == 'Matched') & 
                            (df['Hours_Validation_Status'].str.startswith('Mismatch', na=False))
                        ])
                        # 2. Travel Fee Data Entry Error (Staff recorded fee, Sales did not)
                        travel_fee_error = len(df[
                            (df['Status'] == 'Matched') & 
                            (df['Amount_Status'].str.contains('Staff Log indicates Home Session', na=False))
                        ])
                        # Sum both for the critical mismatch metric
                        error_count = hours_mismatch + travel_fee_error
                    
                    elif pay_type == 'Percentage':
                        # All Amount Mismatch errors are critical for percentage pay staff
                        error_count = len(df[
                            (df['Status'] == 'Matched') & 
                            (df['Amount_Status'].str.startswith('Mismatch', na=False))
                        ])
                    
                    else:
                        error_count = 0
                        
                    return matched_count, error_count, staff_only_count, sales_only_count
                
                h_matched, h_error, h_staff_only, h_sales_only = calculate_metrics(df_hourly, 'Hourly')
                p_matched, p_error, p_staff_only, p_sales_only = calculate_metrics(df_percentage, 'Percentage')

                # --- Display Results ---
                
                st.markdown("### 📊 Reconciliation Summary by Pay Structure")
                
                # Hourly Staff Summary
                st.markdown("#### Hourly Rate Staff Summary")
                col1, col2, col3, col4 = st.columns(4) 
                col1.metric("Total Matches (1-to-1)", h_matched)
                col2.metric("**Critical Mismatch Errors (Hrs + Travel Fee)**", h_error, help="Includes errors where Direct Hours mismatch AND errors where Staff recorded a travel fee but the Sales Subtotal did not.") 
                col3.metric("In Staff Log Only", h_staff_only)
                col4.metric("In Sales Record Only", h_sales_only)

                st.markdown("---")
                
                # Percentage Pay Staff Summary
                st.markdown("#### Percentage Pay Staff Summary")
                col1, col2, col3, col4 = st.columns(4) 
                col1.metric("Total Matches (1-to-1)", p_matched)
                col2.metric("**Amount Mismatch Errors**", p_error, help="Includes general Subtotal mismatches and errors related to missing/mismatched Travel Fees.") 
                col3.metric("In Staff Log Only", p_staff_only)
                col4.metric("In Sales Record Only", p_sales_only)
                
                st.markdown("---")

                st.markdown("### 📋 Detailed Report")
                st.dataframe(final_report, use_container_width=True)

                csv = final_report.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Report as CSV",
                    data=csv,
                    file_name='Reconciliation_Report_Dual_Pay_Travel_Final.csv',
                    mime='text/csv',
                )

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.warning("A critical error prevents processing. Please ensure your CSV files are correctly formatted and uploaded.")

else:
    st.info("👋 Please upload both CSV files in the sidebar to begin.")
