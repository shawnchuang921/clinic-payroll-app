import streamlit as st
import pandas as pd
import sqlite3
import re
import io
import datetime
import numpy as np
from difflib import SequenceMatcher

# --- 1. DATABASE SETUP & MANAGEMENT ---

def init_db():
    """Initializes the SQLite database and handles schema migrations."""
    conn = sqlite3.connect('clinic.db')
    c = conn.cursor()
    
    # Table: Users
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password TEXT,
                    role TEXT,
                    staff_name TEXT
                )''')
    
    # Table: Staff Config (Updated for Indirect Rate)
    c.execute('''CREATE TABLE IF NOT EXISTS staff_config (
                    staff_name TEXT PRIMARY KEY,
                    position TEXT,
                    pay_type TEXT, 
                    base_rate REAL, 
                    travel_fee REAL,
                    indirect_rate REAL
                )''') 
    
    # --- MIGRATION: Check if 'indirect_rate' exists, if not, add it ---
    try:
        c.execute("SELECT indirect_rate FROM staff_config LIMIT 1")
    except sqlite3.OperationalError:
        # Column doesn't exist, add it
        c.execute("ALTER TABLE staff_config ADD COLUMN indirect_rate REAL DEFAULT 0.0")
        conn.commit()

    # Table: Staff Logs
    c.execute('''CREATE TABLE IF NOT EXISTS staff_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    staff_name TEXT,
                    client_name TEXT,
                    direct_hrs REAL,
                    indirect_hrs REAL,
                    charged_amount REAL,
                    outside_clinic TEXT,
                    travel_fee_used REAL,
                    total_pay REAL,
                    notes TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )''')
    
    # --- SEED DATA (Only runs if empty) ---
    c.execute("SELECT count(*) FROM users")
    if c.fetchone()[0] == 0:
        users = [
            ('shawn', 'admin123', 'admin', 'Shawn Chuang'),
            ('leo', 'staff123', 'staff', 'Leonardo Tam'),
            ('julia', 'staff123', 'staff', 'Julia Kwan')
        ]
        c.executemany("INSERT INTO users VALUES (?,?,?,?)", users)
        
        # Default Configs (base_rate is Direct Rate or Percentage Share, indirect_rate is new)
        configs = [
            ('Leonardo Tam', 'OT', 'Percentage', 50.0, 20.0, 0.0), # $50% Share, $20 Travel
            ('Julia Kwan', 'OT', 'Hourly', 80.0, 20.0, 50.0), # $80 Direct, $50 Indirect, $20 Travel
            ('Shawn Chuang', 'Admin', 'Hourly', 0.0, 0.0, 0.0)
        ]
        c.executemany("INSERT INTO staff_config (staff_name, position, pay_type, base_rate, travel_fee, indirect_rate) VALUES (?,?,?,?,?,?)", configs)
        conn.commit()
    
    conn.close()

# --- DB Access Helpers ---

def get_staff_config(staff_name):
    conn = sqlite3.connect('clinic.db')
    # Fetch all columns including the new indirect_rate
    try:
        df = pd.read_sql_query("SELECT * FROM staff_config WHERE staff_name = ?", conn, params=(staff_name,))
    except:
        return None
    conn.close()
    return df.iloc[0] if not df.empty else None

def get_all_staff_names():
    conn = sqlite3.connect('clinic.db')
    df = pd.read_sql_query("SELECT staff_name FROM staff_config", conn)
    conn.close()
    return df['staff_name'].tolist()

def get_log_entry_by_id(log_id):
    conn = sqlite3.connect('clinic.db')
    c = conn.cursor()
    c.execute("SELECT * FROM staff_logs WHERE id=?", (log_id,))
    cols = [desc[0] for desc in c.description]
    row_data = c.fetchone()
    conn.close()
    return dict(zip(cols, row_data)) if row_data else None

def save_log_entry(data):
    conn = sqlite3.connect('clinic.db')
    c = conn.cursor()
    c.execute('''INSERT INTO staff_logs (date, staff_name, client_name, direct_hrs, indirect_hrs, 
                                         charged_amount, outside_clinic, travel_fee_used, total_pay, notes)
                 VALUES (?,?,?,?,?,?,?,?,?,?)''', 
              (data['date'], data['staff_name'], data['client_name'], data['direct_hrs'], 
               data['indirect_hrs'], data['charged_amount'], data['outside_clinic'], 
               data['travel_fee_used'], data['total_pay'], data['notes']))
    conn.commit()
    conn.close()

def update_log_entry(log_id, data):
    conn = sqlite3.connect('clinic.db')
    c = conn.cursor()
    c.execute('''UPDATE staff_logs SET 
                 date=?, client_name=?, direct_hrs=?, indirect_hrs=?, 
                 charged_amount=?, outside_clinic=?, travel_fee_used=?, total_pay=?, notes=?
                 WHERE id=?''', 
              (data['date'], data['client_name'], data['direct_hrs'], 
               data['indirect_hrs'], data['charged_amount'], data['outside_clinic'], 
               data['travel_fee_used'], data['total_pay'], data['notes'], log_id))
    conn.commit()
    conn.close()

def delete_log_entry(log_id):
    conn = sqlite3.connect('clinic.db')
    c = conn.cursor()
    c.execute("DELETE FROM staff_logs WHERE id=?", (log_id,))
    conn.commit()
    conn.close()

def update_staff_info(original_name, new_role, new_pay_type, new_direct_rate, new_indirect_rate, new_travel, new_password=None):
    conn = sqlite3.connect('clinic.db')
    c = conn.cursor()
    # Update Config (base_rate = direct_rate)
    c.execute('''UPDATE staff_config SET 
                 position=?, pay_type=?, base_rate=?, indirect_rate=?, travel_fee=?
                 WHERE staff_name=?''', 
              (new_role, new_pay_type, new_direct_rate, new_indirect_rate, new_travel, original_name))
    
    if new_password:
        c.execute("UPDATE users SET password=? WHERE staff_name=?", (new_password, original_name))
    
    conn.commit()
    conn.close()

def add_new_staff(username, password, staff_name, position, pay_type, direct_rate, indirect_rate, travel_fee):
    """Inserts a new staff member into the users and staff_config tables."""
    conn = sqlite3.connect('clinic.db')
    c = conn.cursor()
    try:
        # 1. Insert into users table
        c.execute("INSERT INTO users VALUES (?,?,?,?)", (username, password, 'staff', staff_name))
        
        # 2. Insert into staff_config table
        c.execute('''INSERT INTO staff_config (staff_name, position, pay_type, base_rate, travel_fee, indirect_rate) 
                     VALUES (?,?,?,?,?,?)''', 
                  (staff_name, position, pay_type, direct_rate, travel_fee, indirect_rate))
        
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        # Handles duplicate username or staff_name (primary keys)
        conn.close()
        return False

def delete_staff(staff_name):
    """Deletes a staff member and all associated records."""
    conn = sqlite3.connect('clinic.db')
    c = conn.cursor()
    
    # 1. Delete from users
    c.execute("DELETE FROM users WHERE staff_name=?", (staff_name,))
    
    # 2. Delete from staff_config
    c.execute("DELETE FROM staff_config WHERE staff_name=?", (staff_name,))
    
    # 3. Delete from staff_logs (to clean up)
    c.execute("DELETE FROM staff_logs WHERE staff_name=?", (staff_name,))
    
    conn.commit()
    conn.close()
    
def get_filtered_logs(staff_filter=None, start_date=None, end_date=None):
    conn = sqlite3.connect('clinic.db')
    query = "SELECT * FROM staff_logs WHERE 1=1"
    params = []
    
    if staff_filter and staff_filter != "All Staff":
        query += " AND staff_name = ?"
        params.append(staff_filter)
    
    if start_date:
        query += " AND date >= ?"
        params.append(str(start_date))
    
    if end_date:
        query += " AND date <= ?"
        params.append(str(end_date))
        
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

# --- PAY CALCULATION HELPER ---

def calculate_total_pay(config, direct_hrs, indirect_hrs, charged_amount, is_home_session):
    direct_rate = config['base_rate']
    indirect_rate = config['indirect_rate']
    travel_fee_val = config['travel_fee'] if is_home_session else 0.0
    total_pay = 0.0
    
    if config['pay_type'] == 'Hourly':
        # Formula: (Direct Hrs × Direct Rate) + (Indirect Hrs × Indirect Rate) + Travel Fee
        total_pay = (direct_hrs * direct_rate) + (indirect_hrs * indirect_rate) + travel_fee_val
    elif config['pay_type'] == 'Percentage':
        # Formula: (Charged Amount × Percentage Share) + Travel Fee
        total_pay = (charged_amount * (direct_rate / 100)) + travel_fee_val
    
    return total_pay, travel_fee_val

# --- 2. RECONCILIATION LOGIC (UNCHANGED) ---

def run_reconciliation_logic(df_staff, df_sales):
    # --- HELPER FUNCTIONS ---
    def clean_name_string(name):
        if not isinstance(name, str): return ""
        return re.sub(r'[^a-z]', '', name.lower())

    def extract_name(note):
        if not isinstance(note, str): return ""
        pattern = r'^(.*?)\s+\d{1,2}(?::\d{2})?-\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)?.*$'
        match = re.search(pattern, note, flags=re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            extracted = re.sub(r'\s+(OT|PT|SLP|Assessment|Intervention|Report|Consultation|Session|Writing)\s*$', '', extracted, flags=re.IGNORECASE)
            return extracted.strip()
        pattern = r'\s+\d{1,2}(?::\d{2})?-\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)?.*$'
        clean_name = re.sub(pattern, '', note, flags=re.IGNORECASE)
        return clean_name.strip()

    def calculate_keyword_score(row):
        if pd.isna(row.get('notes')) or pd.isna(row.get('item')): return 0
        staff_note = str(row['notes']).lower()
        sales_item = str(row['item']).lower()
        keywords = ['report', 'assessment', 'intervention', 'session', 'consultation', 'writing']
        score = 0
        for kw in keywords:
            if kw in staff_note and kw in sales_item: score += 10
        return score

    def extract_expected_hours(item):
        if not isinstance(item, str): return None
        item = item.lower()
        match_min = re.search(r'(\d+)\s*mins?$', item)
        if match_min: return int(match_min.group(1)) / 60.0
        match_hr = re.search(r'(\d+)\s*hours?$', item)
        if match_hr: return float(match_hr.group(1))
        return None

    def check_hours_validation(row):
        if row['Status'] != 'Matched': return 'N/A'
        staff_hrs = row.get('direct_hrs')
        expected_hrs = row.get('expected_hours')
        if pd.isna(staff_hrs) and pd.isna(expected_hrs): return 'Missing Data'
        if pd.isna(staff_hrs) or pd.isna(expected_hrs): return 'Missing Staff/Sales Hrs'
        if round(staff_hrs, 2) == round(expected_hrs, 2): return 'Match'
        return f'Mismatch: Staff Hrs ({staff_hrs}) != Expected Hrs ({expected_hrs})'

    def check_amount_final(row):
        if row['Status'] != 'Matched': return 'N/A'
        staff_charge = row.get('charged_amount', 0)
        travel_fee = row.get('travel_fee_used', 0)
        sales_subtotal = row.get('subtotal', 0)
        expected_hours = row.get('expected_hours', 0)
        outside_clinic = str(row.get('outside_clinic', 'no')).strip().lower()
        
        # Retrieve the staff's specific Direct Rate (merged earlier)
        # Default to 160.0 if missing, but we should have it now.
        staff_direct_rate = row.get('Direct_Rate_Config', 160.0) 
        if pd.isna(staff_direct_rate) or staff_direct_rate == 0:
            staff_direct_rate = 160.0 # Safety fallback
        
        staff_total_charge = staff_charge + travel_fee
        
        # Calculate dynamic base price using the STAFF'S specific direct rate
        expected_base_price = expected_hours * staff_direct_rate
        
        staff_flags_home_session = (outside_clinic == 'yes') and (travel_fee > 0)

        # 1. Primary Check
        if round(staff_total_charge, 2) == round(sales_subtotal, 2):
            if staff_flags_home_session: return 'Match (Inc. Travel Fee)'
            return 'Match'
        
        # 2. Travel Fee Mismatch Scenarios
        if staff_flags_home_session:
            if round(sales_subtotal, 2) == round(expected_base_price, 2):
                return f'Mismatch: Staff Log indicates Home Session (+$20), but Sales Subtotal (${sales_subtotal}) is missing Travel Fee.'
            if round(staff_charge, 2) == 0 and round(sales_subtotal, 2) > 0:
                 return f'Mismatch: Staff Log Charged Amount is $0, but Sales is ${sales_subtotal}. (Possible Travel Fee Error)'

        if not staff_flags_home_session and travel_fee == 0:
            # Check against Staff Rate + 20
            if round(sales_subtotal, 2) == round(expected_base_price + 20.0, 2):
                 return f'Mismatch: Sales Subtotal (${sales_subtotal}) suggests Home Session (+$20) not reflected in Staff Log.'

        return f'Mismatch: Staff Total Charge (${staff_total_charge}) != Sales Subtotal (${sales_subtotal})'

    # --- MAIN PROCESSING ---
    
    # 1. Staff Data Prep
    df_staff.columns = df_staff.columns.str.strip().str.replace(' ', '_').str.lower()
    
    # 2. Fetch Config & Merge Rates
    conn = sqlite3.connect('clinic.db')
    # Fetch base_rate (Direct) and pay_type
    df_config = pd.read_sql_query("SELECT staff_name, pay_type, base_rate FROM staff_config", conn)
    conn.close()
    
    df_config['staff_name_lower'] = df_config['staff_name'].str.lower()
    
    # Create maps
    pay_type_map = df_config.set_index('staff_name_lower')['pay_type'].to_dict()
    rate_map = df_config.set_index('staff_name_lower')['base_rate'].to_dict()
    
    # Merge Info to Staff Log
    df_staff['staff_name_lower'] = df_staff['staff_name'].astype(str).str.lower()
    df_staff['Pay_Type'] = df_staff['staff_name_lower'].map(pay_type_map).fillna('Unknown')
    df_staff['Direct_Rate_Config'] = df_staff['staff_name_lower'].map(rate_map).fillna(160.0)
    df_staff.drop(columns=['staff_name_lower'], inplace=True)

    # 3. Sales Data Prep
    df_sales.columns = df_sales.columns.str.strip().str.replace(' ', '_').str.lower()
    df_sales = df_sales.dropna(subset=['patient', 'invoice_date'])
    df_sales['dt_obj'] = pd.to_datetime(df_sales['invoice_date'], errors='coerce')
    if df_sales['dt_obj'].dt.tz is None:
        df_sales['dt_obj'] = df_sales['dt_obj'].dt.tz_localize('UTC', ambiguous='NaT').dt.tz_convert('America/Vancouver')
    df_sales['date_str'] = df_sales['dt_obj'].dt.normalize().astype(str).str[:10]
    df_sales['patient_norm'] = df_sales['patient'].apply(clean_name_string) 
    df_sales['expected_hours'] = df_sales['item'].apply(extract_expected_hours)
    df_sales['staff_member_lower'] = df_sales['staff_member'].astype(str).str.lower()

    # 4. Standardize Staff Log
    df_staff['date_obj'] = pd.to_datetime(df_staff['date'], errors='coerce')
    df_staff['date_str'] = df_staff['date_obj'].dt.normalize().astype(str).str[:10]
    df_staff['extracted_name'] = df_staff['client_name'] 
    df_staff['name_norm'] = df_staff['extracted_name'].apply(clean_name_string)
    df_staff['travel_fee_used'] = pd.to_numeric(df_staff['travel_fee_used'], errors='coerce').fillna(0)
    df_staff['charged_amount'] = pd.to_numeric(df_staff['charged_amount'], errors='coerce').fillna(0)
    df_staff['direct_hrs'] = pd.to_numeric(df_staff['direct_hrs'], errors='coerce').fillna(0)

    # 5. Matching Logic
    df_staff['staff_id'] = df_staff.index
    df_sales['sales_id'] = df_sales.index

    potential_matches = pd.merge(df_staff, df_sales, left_on=['name_norm'], right_on=['patient_norm'], how='outer', suffixes=('_staff', '_sales'))
    
    def get_date_diff(row):
        if pd.notna(row['date_obj']) and pd.notna(row['dt_obj']):
            return abs((row['date_obj'].date() - row['dt_obj'].date()).days)
        return 999 

    potential_matches['date_diff'] = potential_matches.apply(get_date_diff, axis=1)
    candidates = potential_matches[potential_matches['date_diff'] <= 1].copy()
    candidates['service_score'] = candidates.apply(calculate_keyword_score, axis=1)
    
    def get_match_score(row):
        similarity = SequenceMatcher(None, str(row['name_norm']), str(row['patient_norm'])).ratio()
        score = row['service_score'] + (100 - row['date_diff'])
        if similarity < 0.7: score -= 20 
        return score

    candidates['match_score'] = candidates.apply(get_match_score, axis=1)
    candidates = candidates.sort_values(by='match_score', ascending=False)
    
    matched_staff_ids = set()
    matched_sales_ids = set()
    final_rows = []

    for _, row in candidates.iterrows():
        sid, slid = row['staff_id'], row['sales_id']
        if pd.notna(sid) and pd.notna(slid) and sid not in matched_staff_ids and slid not in matched_sales_ids:
            match_dict = row.to_dict()
            match_dict['Status'] = 'Matched'
            match_dict['Match_Type'] = f"Match (Diff: {row['date_diff']} days)"
            match_dict['Staff_Name_Final'] = row['staff_name'] 
            final_rows.append(match_dict)
            matched_staff_ids.add(sid)
            matched_sales_ids.add(slid)

    # Unmatched Staff
    unmatched_staff = df_staff[~df_staff['staff_id'].isin(matched_staff_ids)].copy()
    for _, row in unmatched_staff.iterrows():
        new_row = row.to_dict()
        if 'date_str' in new_row: new_row['date_str_staff'] = new_row.pop('date_str')
        new_row['Status'], new_row['Match_Type'] = 'In Staff Log Only (Missing in Sales)', 'N/A'
        new_row['Staff_Name_Final'] = row['staff_name']
        new_row.update({'invoice_date': None, 'patient': None, 'item': None, 'subtotal': None, 'expected_hours': None})
        final_rows.append(new_row)

    # Unmatched Sales
    unmatched_sales = df_sales[~df_sales['sales_id'].isin(matched_sales_ids)].copy()
    for _, row in unmatched_sales.iterrows():
        new_row = row.to_dict()
        new_row['date_str_sales'] = new_row.pop('date_str')
        inferred_pay_type = pay_type_map.get(new_row.get('staff_member_lower'), 'Unknown')
        
        # Also grab the rate for the inferred staff to keep columns consistent
        inferred_rate = rate_map.get(new_row.get('staff_member_lower'), 160.0)

        new_row['Staff_Name_Final'] = row['staff_member']
        new_row.update({
            'date_str_staff': None, 'date': None, 'extracted_name': None, 'notes': None, 
            'charged_amount': None, 'direct_hrs': None, 'outside_clinic': None, 
            'travel_fee_used': None, 'Pay_Type': inferred_pay_type, 'Direct_Rate_Config': inferred_rate
        })
        new_row['Status'], new_row['Match_Type'] = 'In Sales Record Only (Missing in Log)', 'N/A'
        final_rows.append(new_row)

    final_df = pd.DataFrame(final_rows)
    final_df['Amount_Status'] = final_df.apply(check_amount_final, axis=1)
    final_df['Hours_Validation_Status'] = final_df.apply(check_hours_validation, axis=1)
    final_df['Display_Date'] = final_df['date_str_staff'].fillna(final_df['date_str_sales'])

    return final_df

# --- 3. PAGE UI FUNCTIONS ---

def login_page():
    st.markdown("## 🔐 Clinic Portal Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        conn = sqlite3.connect('clinic.db')
        c = conn.cursor()
        c.execute("SELECT role, staff_name FROM users WHERE username = ? AND password = ?", (username, password))
        user = c.fetchone()
        conn.close()
        
        if user:
            st.session_state['logged_in'] = True
            st.session_state['user_role'] = user[0]
            st.session_state['staff_name'] = user[1]
            st.rerun()
        else:
            st.error("Invalid Username or Password")

def staff_entry_page():
    st.markdown(f"## 👋 Welcome, {st.session_state['staff_name']}")
    
    config = get_staff_config(st.session_state['staff_name'])
    if config is None:
        st.error("Configuration not found. Please contact Admin.")
        return

    # --- Monthly Pay Estimate (Point 1) ---
    today = datetime.date.today()
    first_day_of_month = today.replace(day=1)
    
    # Fetch all logs for the current staff in the current month
    df_monthly = get_filtered_logs(st.session_state['staff_name'], first_day_of_month, today)
    monthly_pay = df_monthly['total_pay'].sum() if not df_monthly.empty else 0.0

    st.metric(label=f"💰 Estimated Pay for {today.strftime('%B')}", value=f"${monthly_pay:,.2f}")
    st.markdown("---")
    
    # Use 'base_rate' from DB as Direct Rate
    direct_rate = config['base_rate']
    indirect_rate = config['indirect_rate'] if 'indirect_rate' in config else 0.0

    st.info(f"**Position:** {config['position']} | **Pay Type:** {config['pay_type']}")
    if config['pay_type'] == 'Hourly':
        st.caption(f"Rates: ${direct_rate}/hr (Direct) | ${indirect_rate}/hr (Indirect) | Travel Fee: ${config['travel_fee']}")
    elif config['pay_type'] == 'Percentage':
        st.caption(f"Rates: {direct_rate}% of Charged Amt | Travel Fee: ${config['travel_fee']}")

    
    # --- New Entry Form ---
    st.markdown("### 📝 Submit New Entry")
    with st.form("staff_log_form"):
        col1, col2 = st.columns(2)
        date_input = col1.date_input("Date of Service")
        client_name = col2.text_input("Client Name (First Last)")
        col3, col4 = st.columns(2)
        direct_hrs = col3.number_input("Direct Hours", min_value=0.0, step=0.5, key="new_direct_hrs")
        indirect_hrs = col4.number_input("Indirect Hours", min_value=0.0, step=0.5, key="new_indirect_hrs")
        
        charged_amount = 0.0
        if config['pay_type'] == 'Percentage':
            charged_amount = st.number_input("Charged Amount ($)", min_value=0.0, step=10.0, key="new_charged_amt")
        
        is_home_session = st.checkbox("Home Session / Outside Clinic?")
        notes = st.text_area("Notes (Optional)")
        submitted = st.form_submit_button("Submit Entry")
        
        if submitted:
            if not client_name:
                st.error("Client Name is required.")
            else:
                total_pay, travel_fee_val = calculate_total_pay(config, direct_hrs, indirect_hrs, charged_amount, is_home_session)
                outside_val = "Yes" if is_home_session else "No"
                
                log_data = {
                    'date': date_input.strftime('%Y-%m-%d'), 'staff_name': st.session_state['staff_name'],
                    'client_name': client_name, 'direct_hrs': direct_hrs, 'indirect_hrs': indirect_hrs,
                    'charged_amount': charged_amount, 'outside_clinic': outside_val,
                    'travel_fee_used': travel_fee_val, 'total_pay': total_pay, 'notes': f"{client_name} {notes}"
                }
                save_log_entry(log_data)
                st.success(f"Entry Saved! Total Pay: ${total_pay:.2f}")
                st.balloons()
                st.rerun() # Rerun to refresh the monthly total

    # --- View & Edit Logs Section (Points 2 & 3) ---
    st.markdown("### 🔎 View & Edit Your Logs")
    
    # Filters (Point 3)
    col_f1, col_f2, col_f3 = st.columns(3)
    # Default filter to the last 30 days
    today = datetime.date.today()
    default_start_date = today - datetime.timedelta(days=30)
    filter_start_date = col_f1.date_input("Start Date", default_start_date, key="staff_filter_start_date")
    filter_end_date = col_f2.date_input("End Date", today, key="staff_filter_end_date")
    client_search = col_f3.text_input("Client Search (Partial Name)", key="client_search")

    # Fetch filtered data
    df_logs = get_filtered_logs(st.session_state['staff_name'], filter_start_date, filter_end_date)

    # Apply client name filter
    if client_search:
        df_logs = df_logs[df_logs['client_name'].str.contains(client_search, case=False, na=False)]
    
    if df_logs.empty:
        st.info("No entries found for the selected criteria.")
    else:
        # Display Table
        st.dataframe(
            df_logs[['id', 'date', 'client_name', 'direct_hrs', 'indirect_hrs', 'charged_amount', 'outside_clinic', 'total_pay', 'notes']].sort_values('id', ascending=False),
            use_container_width=True
        )

        # Edit/Delete Logic (Point 2)
        st.markdown("#### ✏️ Edit or Delete an Entry")
        log_id_to_edit = st.number_input("Enter Log ID to Edit (Find in 'id' column above)", min_value=0, step=1, key="staff_log_id_edit")
        
        if log_id_to_edit > 0:
            current_row = get_log_entry_by_id(log_id_to_edit)
            
            if current_row and current_row['staff_name'] == st.session_state['staff_name']:
                
                st.warning(f"Editing Log ID: {log_id_to_edit} - Client: {current_row['client_name']}")
                
                with st.form("edit_log_form"):
                    col_e1, col_e2 = st.columns(2)
                    e_date = col_e1.date_input("Date", pd.to_datetime(current_row['date']), key="e_date")
                    e_client = col_e2.text_input("Client Name", current_row['client_name'], key="e_client")
                    
                    col_e3, col_e4 = st.columns(2)
                    e_direct = col_e3.number_input("Direct Hrs", value=float(current_row['direct_hrs']), min_value=0.0, step=0.5, key="e_direct")
                    e_indirect = col_e4.number_input("Indirect Hrs", value=float(current_row['indirect_hrs']), min_value=0.0, step=0.5, key="e_indirect")
                    
                    e_charged = float(current_row['charged_amount'])
                    if config['pay_type'] == 'Percentage':
                        e_charged = st.number_input("Charged Amt", value=float(current_row['charged_amount']), min_value=0.0, step=10.0, key="e_charged")
                    
                    e_is_home = (current_row['outside_clinic'] == 'Yes')
                    e_outside = st.checkbox("Home Session / Outside Clinic?", value=e_is_home, key="e_outside")

                    e_notes = st.text_area("Notes", current_row['notes'], key="e_notes")
                    
                    c_del, c_save = st.columns([1, 4])
                    
                    if c_del.form_submit_button("DELETE Log"):
                        delete_log_entry(log_id_to_edit)
                        st.warning(f"Log {log_id_to_edit} Deleted.")
                        st.rerun()
                        
                    if c_save.form_submit_button("Update Log"):
                        
                        # Recalculate total pay based on potentially updated hours/charged amount
                        recalculated_pay, recalculated_travel = calculate_total_pay(
                            config, e_direct, e_indirect, e_charged, e_outside
                        )

                        updated_data = {
                            'date': e_date.strftime('%Y-%m-%d'), 'client_name': e_client,
                            'direct_hrs': e_direct, 'indirect_hrs': e_indirect,
                            'charged_amount': e_charged, 'outside_clinic': "Yes" if e_outside else "No",
                            'travel_fee_used': recalculated_travel, 'total_pay': recalculated_pay, 'notes': e_notes
                        }
                        update_log_entry(log_id_to_edit, updated_data)
                        st.success(f"Log updated successfully! New Total Pay: ${recalculated_pay:.2f}")
                        st.rerun()

            elif current_row:
                 st.error("You are not authorized to edit this entry.")
            else:
                st.warning("Log ID not found.")
                
def admin_page():
    st.markdown(f"## 🛠️ Admin Dashboard ({st.session_state['staff_name']})")
    
    tab1, tab2, tab3 = st.tabs(["📊 Reconciliation", "👥 Manage Staff & Passwords", "📝 View & Edit Logs (Admin)"])
    
    with tab1:
        st.subheader("Run Payroll Reconciliation")
        
        col1, col2 = st.columns(2)
        all_staff = ["All Staff"] + get_all_staff_names()
        selected_staff = col1.selectbox("Filter by Staff", all_staff)
        
        col3, col4 = col2.columns(2)
        start_date = col3.date_input("Start Date", datetime.date.today().replace(day=1))
        end_date = col4.date_input("End Date", datetime.date.today())
        
        sales_file = st.file_uploader("Upload Sales Record (CSV)", type=['csv'])
        
        if st.button("Run Reconciliation"):
            if not sales_file:
                st.error("Please upload a Sales Record CSV.")
            else:
                df_staff_db = get_filtered_logs(selected_staff, start_date, end_date)
                
                if df_staff_db.empty:
                    st.warning("No staff logs found for the selected criteria.")
                else:
                    try:
                        df_sales_csv = pd.read_csv(sales_file, encoding='latin1', engine='python', on_bad_lines='skip')
                        final_report = run_reconciliation_logic(df_staff_db, df_sales_csv)
                        
                        df_hourly = final_report[final_report['Pay_Type'] == 'Hourly']
                        df_percentage = final_report[final_report['Pay_Type'] == 'Percentage']
                        
                        def get_metrics(df, ptype):
                            matched = len(df[df['Status'] == 'Matched'])
                            staff_only = len(df[df['Status'].str.contains('Staff Log Only', na=False)])
                            sales_only = len(df[df['Status'].str.contains('Sales Record Only', na=False)])
                            err = 0
                            if ptype == 'Hourly':
                                h_err = len(df[(df['Status']=='Matched') & (df['Hours_Validation_Status'].str.startswith('Mismatch'))])
                                amt_err = len(df[(df['Status']=='Matched') & (df['Amount_Status'].str.contains('Mismatch')) & (df['Amount_Status'].str.contains('Travel Fee'))])
                                err = h_err + amt_err
                            else:
                                err = len(df[(df['Status']=='Matched') & (df['Amount_Status'].str.contains('Mismatch'))])
                            return matched, err, staff_only, sales_only

                        h_m, h_e, h_so, h_sro = get_metrics(df_hourly, 'Hourly')
                        p_m, p_e, p_so, p_sro = get_metrics(df_percentage, 'Percentage')
                        
                        st.markdown("#### Hourly Rate Staff Summary")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Matches", h_m)
                        c2.metric("Critical Errors", h_e)
                        c3.metric("Staff Only", h_so)
                        c4.metric("Sales Only", p_sro)
                        
                        st.markdown("#### Percentage Staff Summary")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Matches", p_m)
                        c2.metric("Amount Errors", p_e)
                        c3.metric("Staff Only", p_so)
                        c4.metric("Sales Only", p_sro)
                        
                        st.dataframe(final_report)
                        csv = final_report.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Report", csv, "Reconciliation_Report.csv", "text/csv")
                        
                    except Exception as e:
                        st.error(f"Error processing files: {e}")

    with tab2:
        st.subheader("Manage Staff Information")
        
        # --- ADD STAFF SECTION ---
        st.markdown("### ➕ Add New Staff Member")
        with st.form("add_staff_form"):
            col_n1, col_n2 = st.columns(2)
            new_staff_name = col_n1.text_input("Staff Full Name (Must be Unique)", key="new_staff_name")
            new_login_id = col_n2.text_input("Login ID (Must be Unique)", key="new_login_id")
            
            new_password = st.text_input("Default Password", type="password", key="new_password")
            
            c_r1, c_r2 = st.columns(2)
            new_role = c_r1.text_input("Position/Role", value="OT", key="new_position")
            new_pay_type = c_r2.selectbox("Pay Type", ["Hourly", "Percentage"], key="new_pay_type")

            c_r3, c_r4 = st.columns(2)
            rate_label = "Direct Rate ($/hr or %)"
            new_direct_rate = c_r3.number_input(rate_label, value=80.0, key="new_direct_rate")
            new_indirect_rate = c_r4.number_input("Indirect Rate ($/hr)", value=0.0, key="new_indirect_rate")
            
            new_travel = st.number_input("Travel Fee ($)", value=20.0, key="new_travel")
            
            if st.form_submit_button("Create New Staff"):
                if not new_staff_name or not new_login_id or not new_password:
                    st.error("Please fill in Staff Name, Login ID, and Default Password.")
                else:
                    success = add_new_staff(new_login_id, new_password, new_staff_name, new_role, 
                                            new_pay_type, new_direct_rate, new_indirect_rate, new_travel)
                    if success:
                        st.success(f"Staff member '{new_staff_name}' created successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to add staff. Staff Name or Login ID might already exist.")
        
        st.markdown("---")

        # --- EDIT / DELETE STAFF SECTION ---
        st.markdown("### ✏️ Edit or Remove Existing Staff")
        staff_list = get_all_staff_names()
        selected_staff_edit = st.selectbox("Select Staff to Edit", ["Select..."] + staff_list, key="admin_select_staff_edit")
        
        if selected_staff_edit != "Select...":
            config = get_staff_config(selected_staff_edit)
            
            st.markdown(f"#### Configuration for: **{selected_staff_edit}**")
            
            with st.form("edit_staff_form"):
                c1, c2 = st.columns(2)
                new_role = c1.text_input("Position", value=config['position'])
                new_pay_type = c2.selectbox("Pay Type", ["Hourly", "Percentage"], index=0 if config['pay_type']=='Hourly' else 1)
                
                c3, c4 = st.columns(2)
                rate_label = "Direct Rate ($/hr or %)"
                new_direct_rate = c3.number_input(rate_label, value=config['base_rate'])
                
                indirect_val = config['indirect_rate'] if 'indirect_rate' in config else 0.0
                new_indirect_rate = c4.number_input("Indirect Rate ($/hr)", value=indirect_val)
                
                new_travel = st.number_input("Travel Fee ($)", value=config['travel_fee'])
                
                st.markdown("---")
                st.markdown("**🔐 Security**")
                new_password = st.text_input("New Password (leave blank to keep current)", type="password")
                
                c_del_col, c_update_col = st.columns([1, 4])
                
                # DELETE BUTTON WITH CONFIRMATION
                if c_del_col.button(f"🗑️ Delete Staff"):
                    st.session_state['delete_candidate'] = selected_staff_edit
                    st.session_state['confirm_delete'] = True

                if st.form_submit_button("Update Staff Info"):
                    update_staff_info(selected_staff_edit, new_role, new_pay_type, new_direct_rate, new_indirect_rate, new_travel, new_password if new_password else None)
                    st.success(f"Updated information for {selected_staff_edit}")
                    st.rerun()
                    
            # Confirmation dialogue (outside the form so the form button doesn't trigger a rerun before confirmation)
            if 'confirm_delete' in st.session_state and st.session_state['confirm_delete'] and st.session_state['delete_candidate'] == selected_staff_edit:
                st.error(f"⚠️ Are you absolutely sure you want to delete **{selected_staff_edit}**? This action is permanent and will delete ALL associated logs.")
                col_confirm1, col_confirm2 = st.columns(2)
                if col_confirm1.button("✅ YES, DELETE STAFF PERMANENTLY", key="final_delete_confirm"):
                    delete_staff(selected_staff_edit)
                    st.session_state['confirm_delete'] = False
                    st.session_state['delete_candidate'] = None
                    st.success(f"Staff member {selected_staff_edit} and all their logs have been deleted.")
                    st.rerun()
                if col_confirm2.button("❌ Cancel Deletion", key="cancel_delete"):
                    st.session_state['confirm_delete'] = False
                    st.session_state['delete_candidate'] = None
                    st.info("Deletion cancelled.")

    with tab3:
        st.subheader("View & Edit Staff Logs (Admin Access)")
        
        col1, col2 = st.columns(2)
        filter_staff = col1.selectbox("Filter by Staff", ["All Staff"] + get_all_staff_names(), key="admin_log_filter_staff")
        filter_date = col2.date_input("Filter by Date (Start)", datetime.date.today() - datetime.timedelta(days=30), key="admin_log_filter_date")
        
        df_logs = get_filtered_logs(filter_staff, filter_date, datetime.date.today() + datetime.timedelta(days=1))
        st.dataframe(df_logs)
        
        st.markdown("### ✏️ Edit a Log Entry")
        log_id_to_edit = st.number_input("Enter Log ID to Edit (See 'id' column above)", min_value=0, step=1, key="admin_log_id_edit")
        
        if log_id_to_edit > 0:
            current_row = get_log_entry_by_id(log_id_to_edit)
            
            if current_row:
                staff_config_for_edit = get_staff_config(current_row['staff_name'])
                
                with st.form("admin_edit_log_form"):
                    st.info(f"Editing entry for: **{current_row['staff_name']}**")
                    col_a1, col_a2 = st.columns(2)
                    e_date = col_a1.date_input("Date", pd.to_datetime(current_row['date']), key="a_e_date")
                    e_client = col_a2.text_input("Client Name", current_row['client_name'], key="a_e_client")
                    
                    col_a3, col_a4 = st.columns(2)
                    e_direct = col_a3.number_input("Direct Hrs", value=float(current_row['direct_hrs']), min_value=0.0, step=0.5, key="a_e_direct")
                    e_indirect = col_a4.number_input("Indirect Hrs", value=float(current_row['indirect_hrs']), min_value=0.0, step=0.5, key="a_e_indirect")
                    
                    e_charged = float(current_row['charged_amount'])
                    if staff_config_for_edit['pay_type'] == 'Percentage':
                        e_charged = st.number_input("Charged Amt", value=float(current_row['charged_amount']), min_value=0.0, step=10.0, key="a_e_charged")
                    
                    e_is_home = (current_row['outside_clinic'] == 'Yes')
                    e_outside = st.checkbox("Outside Clinic?", value=e_is_home, key="a_e_outside")
                    
                    e_notes = st.text_area("Notes", current_row['notes'], key="a_e_notes")
                    
                    # Recalculate total pay
                    recalculated_pay, recalculated_travel = calculate_total_pay(
                        staff_config_for_edit, e_direct, e_indirect, e_charged, e_outside
                    )
                    st.write(f"**Recalculated Total Pay:** ${recalculated_pay:.2f} (Travel: ${recalculated_travel:.2f})")
                    
                    c_del, c_save = st.columns([1, 4])
                    if c_del.form_submit_button("DELETE Log (Admin)"):
                        delete_log_entry(log_id_to_edit)
                        st.warning(f"Log {log_id_to_edit} Deleted.")
                        st.rerun()
                        
                    if c_save.form_submit_button("Update Log (Admin)"):
                        updated_data = {
                            'date': e_date.strftime('%Y-%m-%d'), 'client_name': e_client,
                            'direct_hrs': e_direct, 'indirect_hrs': e_indirect,
                            'charged_amount': e_charged, 'outside_clinic': "Yes" if e_outside else "No",
                            'travel_fee_used': recalculated_travel, 'total_pay': recalculated_pay, 'notes': e_notes
                        }
                        update_log_entry(log_id_to_edit, updated_data)
                        st.success("Log updated successfully!")
                        st.rerun()
            else:
                st.warning("Log ID not found.")

# --- 4. APP ENTRY POINT ---

init_db()

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'confirm_delete' not in st.session_state:
    st.session_state['confirm_delete'] = False
if 'delete_candidate' not in st.session_state:
    st.session_state['delete_candidate'] = None

if not st.session_state['logged_in']:
    login_page()
else:
    with st.sidebar:
        st.write(f"Logged in as: **{st.session_state['staff_name']}**")
        if st.button("Logout"):
            st.session_state['logged_in'] = False
            st.session_state['confirm_delete'] = False
            st.session_state['delete_candidate'] = None
            st.rerun()
    
    if st.session_state['user_role'] == 'admin':
        admin_page()
    else:
        staff_entry_page()
