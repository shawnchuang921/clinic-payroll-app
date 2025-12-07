import streamlit as st
import pandas as pd
import re
import io
import numpy as np
from difflib import SequenceMatcher

# --- Page Configuration ---
st.set_page_config(page_title="Clinic Payroll Reconciler", layout="wide")

st.title("🏥 Clinic Staff vs. Sales Reconciler (Robust Column Fix)")
st.markdown("""
The latest code includes fixes for both the timezone bug and the column naming issue ('staff_name_staff'). The final report continues to include the **Staff_Member** column for easy filtering.
""")

# --- Sidebar: File Uploads ---
st.sidebar.header("1. Upload Data")
staff_file = st.sidebar.file_uploader("Upload Combined Staff Log (CSV)", type=['csv'])
sales_file = st.sidebar.file_uploader("Upload Sales Record (CSV)", type=['csv'])

# --- Helper Functions (No changes here, keeping them hidden for brevity) ---
def clean_name_string(name):
    """Aggressively removes all non-alphabetic characters for reliable fuzzy matching."""
    if not isinstance(name, str): return ""
    cleaned = re.sub(r'[^a-z]', '', name.lower())
    return cleaned

def extract_name(note):
    """Clean name from notes by removing time patterns. Tries to capture the client name from the start."""
    if not isinstance(note, str): return ""
    
    # Try to extract the first part of the string before a time range or service description
    pattern = r'^(.*?)\s+\d{1,2}(?::\d{2})?-\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)?.*$'
    match = re.search(pattern, note, flags=re.IGNORECASE)
    
    if match:
        extracted = match.group(1).strip()
        # Further clean by removing common service terms at the end
        extracted = re.sub(r'\s+(OT|PT|SLP|Assessment|Intervention|Report|Consultation|Session|Writing)\s*$', '', extracted, flags=re.IGNORECASE)
        if extracted:
            return extracted.strip()
            
    # Fallback to a simpler cleaning approach
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
    expected_hours = row.get('expected_hours', 0)
    
    # Safely retrieve and standardize Outside_Clinic status
    outside_clinic = str(row.get('outside_clinic', 'no')).strip().lower()
    
    # Calculate the expected total charge from the staff log
    staff_total_charge = staff_charge + travel_fee
    
    # --- Dynamic Base Price Assumption ---
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
        # Check if Sales Subtotal matches the dynamic base price (which implies missing travel fee in sales)
        if round(sales_subtotal, 2) == round(expected_base_price, 2):
            return f'Mismatch: Staff Log indicates Home Session (+$20), but Sales Subtotal (${sales_subtotal}) is missing Travel Fee.'
        
        # Check if Charged_Amount is 0 but sales is non-zero (The Hourly Rate Staff Fix)
        if round(staff_charge, 2) == 0 and round(sales_subtotal, 2) > 0:
             return f'Mismatch: Staff Log Charged Amount is $0, but Sales is ${sales_subtotal}. (Possible Travel Fee Error)'


    # Scenario B: Sales Subtotal suggests a fee, but Staff Log does not reflect it.
    if not staff_flags_home_session and travel_fee == 0:
        # Check if Sales Subtotal matches expected base price + 20
        if round(sales_subtotal, 2) == round(expected_base_price + 20.0, 2):
             return f'Mismatch: Sales Subtotal (${sales_subtotal}) suggests Home Session (+$20) not reflected in Staff Log.'


    # 3. Final Fallback: General Mismatch
    return f'Mismatch: Staff Total Charge (${staff_total_charge}) != Sales Subtotal (${sales_subtotal})'

def get_staff_pay_types(df_staff):
    """
    Categorizes staff as 'Hourly' or 'Percentage' based on their dominant pay calculation column.
    """
    # Standardize column names
    df_staff.columns = df_staff.columns.str.strip().str.replace(' ', '_').str.lower()
    
    numeric_cols = ['direct_pay', 'indirect_pay', 'percentage_pay']
    for col in numeric_cols:
        df_staff[col] = pd.to_numeric(df_staff[col], errors='coerce').fillna(0)
    
    # Assumes 'staff_name' column exists due to robust check/rename in main logic
    staff_pay_totals = df_staff.groupby('staff_name')[numeric_cols].sum()
    
    def determine_pay_type(row):
        hourly_pay = row['direct_pay'] + row['indirect_pay']
        percentage_pay = row['percentage_pay']
        
        if hourly_pay > 0 and percentage_pay == 0:
            return 'Hourly'
        elif percentage_pay > 0 and hourly_pay == 0:
            return 'Percentage'
        elif hourly_pay > 0 and percentage_pay > 0:
            # For mixed pay, prioritize the larger component
            return 'Hourly' if hourly_pay >= percentage_pay else 'Percentage'
        else:
            return 'Unknown'

    staff_pay_types = staff_pay_totals.apply(determine_pay_type, axis=1).reset_index(name='Pay_Type')
    
    # Handle name normalization for staff_name to match sales record's 'staff_member'
    staff_pay_types['staff_name_lower'] = staff_pay_types['staff_name'].astype(str).str.lower()
    
    return staff_pay_types[['staff_name', 'Pay_Type', 'staff_name_lower']]


# --- Run App ---
if staff_file and sales_file:
    st.sidebar.success("Files uploaded successfully!")
    
    if st.button("Run Reconciliation"):
        try:
            with st.spinner('Processing records with Smart Matching and Error Fix...'):
                
                # 1. Load Data
                df_staff = pd.read_csv(staff_file, encoding='latin1', engine='python', on_bad_lines='skip')
                df_sales = pd.read_csv(sales_file, encoding='latin1', engine='python', on_bad_lines='skip')

                # --- PREPROCESSING & CLEANING ---
                
                # Standardize Columns
                df_staff.columns = df_staff.columns.str.strip().str.replace(' ', '_').str.lower()
                df_sales.columns = df_sales.columns.str.strip().str.replace(' ', '_').str.lower()
                
                # NEW FIX: Robust Staff Name Column Check/Rename for consistency before merge
                if 'staff_member' in df_staff.columns and 'staff_name' not in df_staff.columns:
                    # If staff log uses 'staff_member' instead of 'staff_name', rename it for consistency.
                    df_staff.rename(columns={'staff_member': 'staff_name'}, inplace=True)
                elif 'staff_name' not in df_staff.columns:
                    # If 'staff_name' is still missing, we can't proceed.
                    raise ValueError("Staff Log is missing a recognizable 'Staff Name' column ('staff_name' or 'staff_member').")


                # Positional Fix for Staff Date
                if len(df_staff.columns) > 0 and 'date' not in df_staff.columns:
                    df_staff = df_staff.rename(columns={df_staff.columns[0]: 'date'}, errors='ignore')

                # Clean numeric/categorical columns
                df_staff['travel_fee_used'] = pd.to_numeric(df_staff['travel_fee_used'], errors='coerce').fillna(0)
                df_staff['outside_clinic'] = df_staff['outside_clinic'].astype(str).str.strip().str.lower()
                df_staff['charged_amount'] = pd.to_numeric(df_staff['charged_amount'], errors='coerce').fillna(0) 
                
                # Determine Staff Pay Type and Create Pay Type Map
                # This now relies on the renamed 'staff_name' column
                df_pay_types = get_staff_pay_types(df_staff.copy())
                pay_type_map = df_pay_types.set_index('staff_name_lower')['Pay_Type'].to_dict()
                
                # Merge Pay Type onto staff data (for matched/staff-only records)
                df_staff['staff_name_lower'] = df_staff['staff_name'].astype(str).str.lower()
                df_staff = pd.merge(df_staff, df_pay_types[['staff_name_lower', 'Pay_Type']], on='staff_name_lower', how='left')
                df_staff.drop(columns=['staff_name_lower'], inplace=True)
                
                # Create Unique IDs (Crucial for 1-to-1 matching)
                df_staff['staff_id'] = df_staff.index
                df_sales['sales_id'] = df_sales.index

                # Process Sales Dates & Names
                df_sales = df_sales.dropna(subset=['patient', 'invoice_date'])
                df_sales['dt_obj'] = pd.to_datetime(df_sales['invoice_date'], errors='coerce')
                
                # FIX 1: Timezone error fix
                if df_sales['dt_obj'].dt.tz is None:
                    # Use ambiguous='NaT' for safety, removing the invalid 'errors' argument
                    df_sales['dt_obj'] = df_sales['dt_obj'].dt.tz_localize('UTC', ambiguous='NaT').dt.tz_convert('America/Vancouver')
                    
                df_sales['date_str'] = df_sales['dt_obj'].dt.normalize().astype(str).str[:10]
                df_sales['patient_norm'] = df_sales['patient'].apply(clean_name_string) 
                df_sales['expected_hours'] = df_sales['item'].apply(extract_expected_hours)
                df_sales['staff_member_lower'] = df_sales['staff_member'].astype(str).str.lower()
                
                # Process Staff Dates & Names
                df_staff['date_obj'] = pd.to_datetime(df_staff['date'], errors='coerce')
                df_staff['date_str'] = df_staff['date_obj'].dt.normalize().astype(str).str[:10]
                df_staff['extracted_name'] = df_staff['notes'].apply(extract_name)
                df_staff['name_norm'] = df_staff['extracted_name'].apply(clean_name_string)

                # --- MATCHING LOGIC ---
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
                
                # Use a composite score
                def get_match_score(row):
                    similarity = SequenceMatcher(None, row['name_norm'], row['patient_norm']).ratio()
                    score = row['service_score'] + (100 - row['date_diff'])
                    if similarity < 0.7:
                        score -= 20 
                    return score

                candidates['match_score'] = candidates.apply(get_match_score, axis=1)
                candidates = candidates.sort_values(by='match_score', ascending=False)
                
                matched_staff_ids = set()
                matched_sales_ids = set()
                final_rows = []

                # Process Matched Records
                for _, row in candidates.iterrows():
                    sid, slid = row['staff_id'], row['sales_id']
                    if pd.notna(sid) and pd.notna(slid) and sid not in matched_staff_ids and slid not in matched_sales_ids:
                        match_dict = row.to_dict()
                        match_dict['Status'] = 'Matched'
                        match_dict['Match_Type'] = f"Match (Diff: {row['date_diff']} days)"
                        # Staff Name is from the staff log (staff_name_staff) because we renamed the column earlier
                        match_dict['Staff_Name_Final'] = row['staff_name_staff'] 
                        final_rows.append(match_dict)
                        matched_staff_ids.add(sid)
                        matched_sales_ids.add(slid)

                # Process Unmatched Staff Records 
                unmatched_staff = df_staff[~df_staff['staff_id'].isin(matched_staff_ids)].copy()
                for _, row in unmatched_staff.iterrows():
                    new_row = row.to_dict()
                    if 'date_str' in new_row: new_row['date_str_staff'] = new_row.pop('date_str')
                    new_row['Status'], new_row['Match_Type'] = 'In Staff Log Only (Missing in Sales)', 'N/A'
                    # Staff Name is from the staff log's 'staff_name' column
                    new_row['Staff_Name_Final'] = row['staff_name'] 
                    # Populate necessary sales columns with None/NaN
                    new_row.update({'invoice_date': None, 'patient': None, 'item': None, 'subtotal': None, 'expected_hours': None, 'staff_member': None})
                    final_rows.append(new_row)

                # Process Unmatched Sales Records
                unmatched_sales = df_sales[~df_sales['sales_id'].isin(matched_sales_ids)].copy()
                for _, row in unmatched_sales.iterrows():
                    new_row = row.to_dict()
                    new_row['date_str_sales'] = new_row.pop('date_str')
                    
                    sales_staff_lower = new_row.get('staff_member_lower')
                    inferred_pay_type = pay_type_map.get(sales_staff_lower, 'Unknown')
                    
                    # Staff Name is from the sales record's 'staff_member' column
                    new_row['Staff_Name_Final'] = row['staff_member']

                    # Populate necessary staff columns with None/NaN
                    new_row.update({
                        'date_str_staff': None, 'date': None, 'extracted_name': None, 'notes': None, 
                        'charged_amount': None, 'direct_hrs': None, 'outside_clinic': None, 
                        'travel_fee_used': None, 'Pay_Type': inferred_pay_type 
                    })
                    new_row['Status'], new_row['Match_Type'] = 'In Sales Record Only (Missing in Log)', 'N/A'
                    final_rows.append(new_row)

                final_df = pd.DataFrame(final_rows)
                
                # --- FINAL REPORT VALIDATION & FORMATTING ---
                final_df['Amount_Status'] = final_df.apply(check_amount_final, axis=1)
                final_df['Hours_Validation_Status'] = final_df.apply(check_hours_validation, axis=1)
                final_df['Display_Date'] = final_df['date_str_staff'].fillna(final_df['date_str_sales'])

                # Column order with new Staff_Member column first
                report_cols = ['Display_Date', 'Staff_Name_Final', 'Pay_Type', 'extracted_name', 'notes', 
                               'outside_clinic', 'travel_fee_used', 'direct_hrs', 'charged_amount', 
                               'invoice_date', 'patient', 'item', 'expected_hours', 'subtotal', 
                               'Status', 'Amount_Status', 'Hours_Validation_Status', 'Match_Type']
                
                rename_map = {
                    'Display_Date': 'Date (Reconciled)', 
                    'Staff_Name_Final': 'Staff_Member', 
                    'extracted_name': 'Client_Name (Staff Log)',
                    'Pay_Type': 'Staff_Pay_Type',
                    'notes': 'Staff_Notes', 
                    'outside_clinic': 'Staff_Outside_Clinic',
                    'travel_fee_used': 'Staff_Travel_Fee',
                    'direct_hrs': 'Staff_Direct_Hrs', 
                    'charged_amount': 'Charged_Amount',
                    'invoice_date': 'Invoice Date', 
                    'patient': 'Client_Name (Sales)',
                    'item': 'Sales_Item', 
                    'expected_hours': 'Sales_Expected_Hrs', 
                    'subtotal': 'Sales_Subtotal',
                    'Match_Type': 'Date_Tolerance'
                }

                available_cols = [c for c in report_cols if c in final_df.columns]
                final_report = final_df[available_cols].rename(columns=rename_map)

                # --- METRICS CALCULATION (omitted for brevity) ---
                df_hourly = final_report[final_report['Staff_Pay_Type'] == 'Hourly'].copy()
                df_percentage = final_report[final_report['Staff_Pay_Type'] == 'Percentage'].copy()
                
                def calculate_metrics(df, pay_type):
                    matched_count = len(df[df['Status'] == 'Matched'])
                    staff_only_count = len(df[df['Status'] == 'In Staff Log Only (Missing in Sales)'])
                    sales_only_count = len(df[df['Status'] == 'In Sales Record Only (Missing in Log)'])
                    
                    if pay_type == 'Hourly':
                        hours_mismatch = len(df[(df['Status'] == 'Matched') & (df['Hours_Validation_Status'].str.startswith('Mismatch', na=False))])
                        amount_mismatch = len(df[(df['Status'] == 'Matched') & (df['Amount_Status'].str.contains('Mismatch', na=False))])
                        error_count = hours_mismatch + amount_mismatch
                    
                    elif pay_type == 'Percentage':
                        error_count = len(df[(df['Status'] == 'Matched') & (df['Amount_Status'].str.startswith('Mismatch', na=False))])
                    
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
                col2.metric("**Critical Mismatch Errors (Hrs + Amount)**", h_error, help="Includes errors where Direct Hours mismatch AND general amount mismatches (including travel fee discrepancies).") 
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

                st.markdown("### 📋 Detailed Report (Includes Staff Member Name for Filtering)")
                st.dataframe(final_report, use_container_width=True)

                csv = final_report.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Report as CSV",
                    data=csv,
                    file_name='Reconciliation_Report_Fixed_Errors_and_Names.csv',
                    mime='text/csv',
                )

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.warning("A critical error prevents processing. Please ensure your CSV files are correctly formatted and uploaded.")
            # For debugging, you could print the list of columns to help the user check their file structure
            if staff_file and 'df_staff' in locals():
                st.info(f"Staff Log Columns Found: {list(df_staff.columns)}")
            if sales_file and 'df_sales' in locals():
                st.info(f"Sales Record Columns Found: {list(df_sales.columns)}")

else:
    st.info("👋 Please upload both CSV files in the sidebar to begin.")
