import streamlit as st
import pandas as pd
import re
import io

# --- Page Configuration ---
st.set_page_config(page_title="Clinic Payroll Reconciler", layout="wide")

st.title("🏥 Clinic Staff vs. Sales Reconciler")
st.markdown("""
Upload the **Staff Log** and the **Sales Record** below. 
The app uses **Smart 1-to-1 Matching** to align records based on Name, Date, and Service Type.
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

def calculate_keyword_score(row):
    """
    Gives a bonus score if the Staff Notes and Sales Item share key service words.
    """
    # FIX: Use lowercase keys 'notes' and 'item' because we standardized columns earlier
    if pd.isna(row.get('notes')) or pd.isna(row.get('item')):
        return 0
    
    staff_note = str(row['notes']).lower()
    sales_item = str(row['item']).lower()
    
    keywords = ['report', 'assessment', 'intervention', 'session', 'consultation', 'writing']
    score = 0
    
    # Check for keyword overlap
    for kw in keywords:
        if kw in staff_note and kw in sales_item:
            score += 10 # 10 points for a service type match
            
    return score

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

                # Create Unique IDs (Crucial for 1-to-1 matching)
                df_staff['staff_id'] = df_staff.index
                df_sales['sales_id'] = df_sales.index

                # Process Sales Dates & Names
                df_sales = df_sales.dropna(subset=['patient', 'invoice_date'])
                df_sales['dt_obj'] = pd.to_datetime(df_sales['invoice_date'], utc=True)
                df_sales['dt_local'] = df_sales['dt_obj'].dt.tz_convert('America/Vancouver')
                df_sales['date_str'] = df_sales['dt_local'].dt.normalize().astype(str).str[:10]
                df_sales['patient_norm'] = df_sales['patient'].apply(clean_name_string) 

                # Process Staff Dates & Names
                df_staff['date_obj'] = pd.to_datetime(df_staff['date'])
                df_staff['date_str'] = df_staff['date_obj'].dt.normalize().astype(str).str[:10]
                df_staff['extracted_name'] = df_staff['notes'].apply(extract_name)
                df_staff['name_norm'] = df_staff['extracted_name'].apply(clean_name_string)

                # --- MATCHING LOGIC ---
                
                # 1. Outer Merge on Name Only (Get all potential candidates)
                potential_matches = pd.merge(
                    df_staff,
                    df_sales,
                    left_on=['name_norm'],
                    right_on=['patient_norm'],
                    how='outer',
                    suffixes=('_staff', '_sales')
                )
                
                # 2. Filter Candidates by Date Tolerance (must be within 1 day)
                # Helper to calc date diff
                def get_date_diff(row):
                    if pd.notna(row['date_obj']) and pd.notna(row['dt_obj']):
                        return abs((row['date_obj'].date() - row['dt_obj'].date()).days)
                    return 999 # High number if no date match

                potential_matches['date_diff'] = potential_matches.apply(get_date_diff, axis=1)
                
                # Keep only matches within 1 day, OR rows that failed to match (NaNs)
                candidates = potential_matches[potential_matches['date_diff'] <= 1].copy()
                
                # 3. Score the Candidates (Keyword Match)
                candidates['service_score'] = candidates.apply(calculate_keyword_score, axis=1)
                
                # 4. GREEDY ASSIGNMENT (The 1-to-1 Logic)
                # Sort candidates by: 
                #   1. Service Score (Highest first - prefer "Report" matches "Report")
                #   2. Date Diff (Lowest first - prefer 0 days over 1 day)
                candidates = candidates.sort_values(by=['service_score', 'date_diff'], ascending=[False, True])
                
                matched_staff_ids = set()
                matched_sales_ids = set()
                final_rows = []

                # Iterate through sorted candidates and lock in matches
                for _, row in candidates.iterrows():
                    sid = row['staff_id']
                    slid = row['sales_id']
                    
                    # If both this staff record and sales record are still free, match them!
                    if sid not in matched_staff_ids and slid not in matched_sales_ids:
                        row['Status'] = 'Matched'
                        row['Match_Type'] = f"Match (Diff: {row['date_diff']} days)"
                        final_rows.append(row)
                        matched_staff_ids.add(sid)
                        matched_sales_ids.add(slid)

                # 5. Handle Unmatched Records
                
                # Find Staff records that were NOT matched
                unmatched_staff = df_staff[~df_staff['staff_id'].isin(matched_staff_ids)].copy()
                for _, row in unmatched_staff.iterrows():
                    new_row = row.to_dict()
                    new_row['Status'] = 'In Staff Log Only (Missing in Sales)'
                    new_row['Match_Type'] = 'N/A'
                    # Fill missing sales columns with NaN
                    new_row['invoice_date'] = None
                    new_row['patient'] = None
                    new_row['item'] = None
                    new_row['subtotal'] = None
                    final_rows.append(new_row)

                # Find Sales records that were NOT matched
                unmatched_sales = df_sales[~df_sales['sales_id'].isin(matched_sales_ids)].copy()
                for _, row in unmatched_sales.iterrows():
                    new_row = row.to_dict()
                    new_row['Status'] = 'In Sales Record Only (Missing in Log)'
                    new_row['Match_Type'] = 'N/A'
                    # Fill missing staff columns with NaN
                    new_row['date'] = None
                    new_row['extracted_name'] = None
                    new_row['notes'] = None
                    new_row['charged_amount'] = None
                    final_rows.append(new_row)

                # Create Final DataFrame
                final_df = pd.DataFrame(final_rows)
                
                # --- FINAL REPORT FORMATTING ---
                
                def check_amount_final(row):
                    if row['Status'] == 'Matched':
                        if row.get('charged_amount') == row.get('subtotal'):
                            return 'Match'
                        else:
                            return 'Mismatch'
                    return 'N/A'

                final_df['Amount_Status'] = final_df.apply(check_amount_final, axis=1)
                
                # Consolidate Date Column
                final_df['Display_Date'] = final_df['date'].fillna(final_df['date_str'])

                report_cols = ['Display_Date', 'extracted_name', 'notes', 'charged_amount',
                               'invoice_date', 'patient', 'item', 'subtotal', 
                               'Status', 'Amount_Status', 'Match_Type']
                
                # Rename for display
                rename_map = {
                    'Display_Date': 'Date (Staff Log)', 
                    'extracted_name': 'Staff_Patient_Name',
                    'notes': 'Notes', 
                    'charged_amount': 'Charged_Amount',
                    'invoice_date': 'Invoice Date', 
                    'patient': 'Patient',
                    'item': 'Item', 
                    'subtotal': 'Subtotal',
                    'Match_Type': 'Date_Tolerance'
                }

                final_report = final_df[report_cols].rename(columns=rename_map)

                # --- METRICS ---
                matched_count = len(final_report[final_report['Status'] == 'Matched'])
                staff_only_count = len(final_report[final_report['Status'] == 'In Staff Log Only (Missing in Sales)'])
                sales_only_count = len(final_report[final_report['Status'] == 'In Sales Record Only (Missing in Log)'])

                # --- Display Results ---
                st.markdown("### 📊 Reconciliation Summary")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Matches (1-to-1)", matched_count)
                col2.metric("In Staff Log Only", staff_only_count)
                col3.metric("In Sales Record Only", sales_only_count)

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
