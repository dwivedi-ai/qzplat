# stereotype_quiz_app/app.py
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# WARNING: HARDCODED CREDENTIALS BELOW - EXTREME SECURITY RISK
# USE ONLY AS A TEMPORARY URGENT FIX. DO NOT COMMIT TO VERSION CONTROL.
# REPLACE PLACEHOLDERS WITH YOUR ACTUAL VALUES.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import os
import csv
import io
import traceback
# from urllib.parse import urlparse # No longer needed for parsing env var

import mysql.connector
from mysql.connector import Error as MySQLError
import pandas as pd
import numpy as np

# Ensure Flask components are imported
from flask import (
    Flask, render_template, request, redirect, url_for, g,
    flash, Response, send_file, get_flashed_messages
)

# --- Configuration Settings ---
CSV_FILE_PATH = os.path.join('data', 'stereotypes.csv') # Relative path to CSV
SCHEMA_FILE = 'schema_mysql.sql'      # Relative path to schema file
RESULTS_TABLE = 'results'             # Name of the table in the database

# --- BEGIN HARDCODED CONFIGURATION (DANGEROUS!) ---

# 1. SECRET KEY - Generate a real one and paste it here
#    Run in terminal: python -c 'import os; import binascii; print(binascii.hexlify(os.urandom(24)).decode("utf-8"))'
HARDCODED_SECRET_KEY = "respai" # Replace this placeholder

# 2. DATABASE CONNECTION DETAILS - Replace placeholders with your actual Railway MySQL values
HARDCODED_DB_CONFIG = {
    'MYSQL_HOST': "mysql.railway.internal", # e.g., "mysql.railway.internal" or similar from MySQL service env vars
    'MYSQL_USER': "root",                                # From MySQL service env vars
    'MYSQL_PASSWORD': "KmWrKJtaEiobgeqshiIrwzMaIQmEXAPn", # Your actual root password from MySQL service env vars
    'MYSQL_DB': "railway",                               # From MySQL service env vars
    'MYSQL_PORT': 3306                                   # Default MySQL port, confirm from MySQL service env vars if different
}

print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("WARNING: Using HARDCODED database credentials in app.py!")
print("THIS IS A MAJOR SECURITY RISK AND SHOULD BE TEMPORARY.")
print("Host:", HARDCODED_DB_CONFIG['MYSQL_HOST'])
print("Database:", HARDCODED_DB_CONFIG['MYSQL_DB'])
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

# Basic validation of hardcoded config
is_db_configured = False
if not all([HARDCODED_DB_CONFIG.get('MYSQL_HOST'),
            HARDCODED_DB_CONFIG.get('MYSQL_USER'),
            HARDCODED_DB_CONFIG.get('MYSQL_PASSWORD'), # Password is required now
            HARDCODED_DB_CONFIG.get('MYSQL_DB')]):
    print("CRITICAL ERROR: Hardcoded database configuration is incomplete. Check placeholders.")
else:
    is_db_configured = True
    print("--- Hardcoded database configuration appears complete. ---")


# --- Flask App Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = HARDCODED_SECRET_KEY

# Populate app.config ONLY if hardcoded config seems complete
if is_db_configured:
    app.config.update(HARDCODED_DB_CONFIG) # Add MYSQL_HOST, MYSQL_USER etc. to app.config
    app.config['DB_CONFIGURED'] = True
    print("Flask app configured with HARDCODED database details.")
else:
    app.config['DB_CONFIGURED'] = False
    print("Flask app database NOT configured due to incomplete HARDCODED details.")

# --- END HARDCODED CONFIGURATION ---


# --- Database Functions (These now read from app.config populated by hardcoded values) ---
def get_db():
    """Opens a new MySQL connection and cursor for the current request context."""
    if 'db' not in g:
        if not app.config.get('DB_CONFIGURED', False):
            print("ERROR get_db: Attempted to get DB connection, but DB is not configured (Hardcoded issue?).")
            flash('Database is not configured. Please contact the administrator.', 'error')
            g.db = None
            g.cursor = None
            return None

        try:
            # Connect using details from app.config (which came from hardcoded dict)
            g.db = mysql.connector.connect(
                host=app.config['MYSQL_HOST'],
                user=app.config['MYSQL_USER'],
                password=app.config['MYSQL_PASSWORD'], # Password must exist now
                database=app.config['MYSQL_DB'],
                port=app.config.get('MYSQL_PORT', 3306),
                connection_timeout=10
            )
            g.cursor = g.db.cursor(dictionary=True)
        except MySQLError as err:
            print(f"ERROR get_db: MySQL connection failed: {err}")
            print(f"DEBUG: Connection attempt failed with Host={app.config.get('MYSQL_HOST')}, User={app.config.get('MYSQL_USER')}, DB={app.config.get('MYSQL_DB')}")
            flash('Database connection error. Please try again later.', 'error')
            g.db = None; g.cursor = None
        except Exception as e:
            print(f"UNEXPECTED ERROR in get_db during connect: {e}\n{traceback.format_exc()}")
            flash('An unexpected error occurred connecting to the database.', 'error')
            g.db = None; g.cursor = None

    return getattr(g, 'cursor', None)

@app.teardown_appcontext
def close_db(error):
    """Closes the database connection and cursor at the end of the request."""
    cursor = g.pop('cursor', None); db = g.pop('db', None)
    if cursor:
        try: cursor.close()
        except Exception as e: print(f"Error closing cursor: {e}")
    if db and db.is_connected():
        try: db.close()
        except Exception as e: print(f"Error closing DB connection: {e}")
    if error: print(f"App context teardown error detected: {error}")

def init_db():
    """Initializes the database using hardcoded config."""
    if not app.config.get('DB_CONFIGURED'):
        print("CRITICAL init_db: Skipping DB initialization because DB is not configured (Hardcoded issue?).")
        return

    # Use connection details directly from app.config (populated by hardcoded dict)
    db_host = app.config['MYSQL_HOST']
    db_user = app.config['MYSQL_USER']
    db_password = app.config['MYSQL_PASSWORD']
    db_port = app.config.get('MYSQL_PORT', 3306)
    db_name = app.config['MYSQL_DB']

    print(f"--- init_db: Starting Initialization Check (Using Hardcoded Config) ---")
    print(f"  Target DB: {db_name} on {db_host}:{db_port}")

    temp_conn = None; temp_cursor = None
    try:
        # 1. Connect to MySQL server
        print("init_db: Connecting to MySQL server...")
        temp_conn = mysql.connector.connect(
            host=db_host, user=db_user, password=db_password, port=db_port, connection_timeout=15
        )
        temp_cursor = temp_conn.cursor()
        print("init_db: Connected to server.")

        # 2. Create database if it doesn't exist
        try:
            temp_cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            temp_conn.commit()
            print(f"init_db: Database '{db_name}' checked/created (if permissions allowed).")
        except MySQLError as err:
            print(f"Warning init_db: Could not execute CREATE DATABASE IF NOT EXISTS for '{db_name}'. Error: {err}")

        # 3. Select the database
        try:
            temp_cursor.execute(f"USE `{db_name}`")
            print(f"init_db: Successfully selected database '{db_name}'.")
        except MySQLError as err:
            print(f"CRITICAL init_db ERROR: Failed to select database '{db_name}'. Error: {err}")
            return # Stop initialization if we can't USE the database

        # 4. Check if the results table exists
        print(f"init_db: Checking if table '{RESULTS_TABLE}' exists...")
        temp_cursor.execute(f"SHOW TABLES LIKE '{RESULTS_TABLE}'")
        table_exists = temp_cursor.fetchone()

        # 5. If table doesn't exist, execute the schema file
        if not table_exists:
            print(f"init_db: Table '{RESULTS_TABLE}' not found. Executing schema from '{SCHEMA_FILE}'...")
            schema_path = os.path.join(app.root_path, SCHEMA_FILE)
            # Add fallback check for schema location
            if not os.path.exists(schema_path):
                schema_path_alt = os.path.join(app.root_path, 'schema_mysql.sql')
                if os.path.exists(schema_path_alt): schema_path = schema_path_alt
                else:
                    print(f"CRITICAL init_db ERROR: Schema file '{SCHEMA_FILE}' not found at '{schema_path}' or '{schema_path_alt}'.")
                    raise FileNotFoundError(f"Schema file missing: {SCHEMA_FILE}")

            try:
                with open(schema_path, mode='r', encoding='utf-8') as f:
                    sql_script = f.read()
                print(f"init_db: Executing SQL script from {schema_path}...")
                statement_results = temp_cursor.execute(sql_script, multi=True)
                for result in statement_results: # Consume results
                    if result.with_rows: result.fetchall()
                temp_conn.commit()
                print(f"init_db: Schema executed successfully. Table '{RESULTS_TABLE}' should now exist.")
            except Exception as e_schema: # Catch broad errors during schema execution
                 print(f"CRITICAL init_db ERROR executing schema: {e_schema}\n{traceback.format_exc()}")
                 try: temp_conn.rollback()
                 except Exception as rb_err: print(f"Rollback failed: {rb_err}")

        else:
            print(f"init_db: Table '{RESULTS_TABLE}' already exists.")

    except MySQLError as e:
        print(f"CRITICAL init_db ERROR during connection/setup phase: {e}")
    except Exception as e:
        print(f"CRITICAL init_db UNEXPECTED error during initialization: {e}\n{traceback.format_exc()}")
    finally:
        if temp_cursor:
            try: temp_cursor.close()
            except Exception as e_close: print(f"Warning: Error closing init_db cursor: {e_close}")
        if temp_conn and temp_conn.is_connected():
            try: temp_conn.close()
            except Exception as e_close: print(f"Warning: Error closing init_db connection: {e_close}")
        print("--- init_db: Finished Initialization Check ---")


# --- Initialize DB on Application Start ---
if app.config.get('DB_CONFIGURED'):
    print(">>> Application starting: Performing database initialization check (using hardcoded config)...")
    with app.app_context(): init_db()
    print(">>> Application starting: Database initialization check complete.")
else:
    print(">>> Application starting: Skipping DB initialization check (Hardcoded DB config incomplete).")


# --- Data Loading Function (Load Stereotypes from CSV) ---
# (Keep load_stereotype_data exactly as before)
def load_stereotype_data(relative_filepath=CSV_FILE_PATH):
    stereotype_data = []; full_filepath = os.path.join(app.root_path, relative_filepath)
    print(f"--- load_stereotype_data: Attempting to load from: {full_filepath} ---")
    try:
        if not os.path.exists(full_filepath): raise FileNotFoundError(f"Not found: {full_filepath}")
        with open(full_filepath, mode='r', encoding='utf-8-sig') as infile:
            reader = csv.DictReader(infile); required_cols = ['State', 'Category', 'Superset', 'Subsets']
            if not reader.fieldnames or not all(f in reader.fieldnames for f in required_cols):
                 raise ValueError(f"CSV missing required cols. Found: {reader.fieldnames}")
            row_count, error_count = 0, 0; processed_count = 0
            for i, row in enumerate(reader):
                row_count += 1; state = row.get('State','').strip(); category = row.get('Category','').strip() or 'Uncategorized'
                superset = row.get('Superset','').strip(); subsets_str = row.get('Subsets','')
                try:
                    if not state or not superset: error_count += 1; continue
                    subsets = sorted([s.strip() for s in subsets_str.split(',') if s.strip()])
                    stereotype_data.append({'state': state,'category': category,'superset': superset,'subsets': subsets})
                    processed_count +=1
                except Exception as row_err: print(f"Err row {i+2}: {row_err}"); error_count += 1; continue
        print(f"--- load_stereotype_data: Successfully loaded {processed_count} entries (skipped {error_count}). ---")
        return stereotype_data
    except FileNotFoundError as e: print(f"FATAL: Stereotype file not found: {e}"); return []
    except ValueError as e: print(f"FATAL: CSV format error: {e}"); return []
    except Exception as e: print(f"FATAL loading stereotypes: {e}\n{traceback.format_exc()}"); return []


# --- Load Data & States on App Start ---
print(">>> Loading stereotype definitions...")
ALL_STEREOTYPE_DATA = load_stereotype_data()
INDIAN_STATES = sorted(list(set(item['state'] for item in ALL_STEREOTYPE_DATA))) if ALL_STEREOTYPE_DATA else []
if not ALL_STEREOTYPE_DATA or not INDIAN_STATES:
    print("CRITICAL WARNING: Stereotype data loading failed/empty.")
    INDIAN_STATES = ["Error: State data unavailable"]
else: print(f">>> Stereotype definitions loaded for {len(INDIAN_STATES)} states.")


# --- Data Processing Logic (for Admin Downloads) ---
# (Keep calculate_mean_offensiveness as before)
def calculate_mean_offensiveness(series):
    valid_ratings = series[series >= 0]; return valid_ratings.mean() if not valid_ratings.empty else np.nan

# (generate_aggregated_data now uses hardcoded config via app.config)
def generate_aggregated_data():
    print("--- generate_aggregated_data: Starting ---")
    aggregated_df = pd.DataFrame() # Default to empty DataFrame

    if not app.config.get('DB_CONFIGURED'):
        print("ERROR generate_aggregated_data: Database not configured (Hardcoded issue?).")
        flash("Cannot generate data: Database is not configured.", "error")
        return None # Return None to indicate failure clearly

    db_conn_proc = None
    try:
        print("generate_aggregated_data: Connecting to DB (using hardcoded config)...")
        db_conn_proc = mysql.connector.connect(
            host=app.config['MYSQL_HOST'], user=app.config['MYSQL_USER'],
            password=app.config['MYSQL_PASSWORD'], database=app.config['MYSQL_DB'],
            port=app.config.get('MYSQL_PORT', 3306), connection_timeout=15
        )
        if not db_conn_proc.is_connected():
             raise MySQLError("Processing: Failed to establish database connection.")
        print("generate_aggregated_data: Connected. Fetching raw results...")

        # --- REST OF generate_aggregated_data REMAINS THE SAME ---
        # (Fetch raw results, load definitions, process, aggregate)
        results_df = pd.read_sql_query(f"SELECT * FROM {RESULTS_TABLE}", db_conn_proc)
        print(f"generate_aggregated_data: Fetched {len(results_df)} raw results.")
        if results_df.empty: return aggregated_df

        required_result_cols = ['user_state', 'category', 'attribute_superset', 'annotation', 'offensiveness_rating']
        if not all(col in results_df.columns for col in required_result_cols):
             missing_cols = [col for col in required_result_cols if col not in results_df.columns]
             raise ValueError(f"Database results table is missing required columns: {missing_cols}")

        results_df['Stereotype_State'] = results_df['user_state']
        if not ALL_STEREOTYPE_DATA: raise ValueError("Stereotype definitions were not loaded correctly earlier.")

        subset_lookup = {}
        for item in ALL_STEREOTYPE_DATA:
            subset_lookup[(item['state'], item['category'], item['superset'])] = item['subsets']

        print("generate_aggregated_data: Expanding results based on subsets...")
        expanded_rows = []; processing_errors = 0
        for index, result_row in results_df.iterrows():
            try:
                state = result_row.get('Stereotype_State'); category = result_row.get('category'); superset = result_row.get('attribute_superset')
                annotation = result_row.get('annotation'); rating_val = result_row.get('offensiveness_rating'); rating = int(rating_val) if pd.notna(rating_val) else -1
                if not all([state, category, superset, annotation is not None]): processing_errors += 1; continue
                expanded_rows.append({'Stereotype_State': state, 'Category': category, 'Attribute': superset, 'annotation': annotation, 'offensiveness_rating': rating})
                subsets_list = subset_lookup.get((state, category, superset), [])
                for subset in subsets_list: expanded_rows.append({'Stereotype_State': state, 'Category': category, 'Attribute': subset, 'annotation': annotation, 'offensiveness_rating': rating})
            except Exception as row_proc_err: processing_errors += 1; print(f"Error processing result row {index}: {row_proc_err}")
        if processing_errors > 0: print(f"Warning generate_aggregated_data: Skipped {processing_errors} rows during expansion.")

        if not expanded_rows: return aggregated_df
        expanded_annotations_df = pd.DataFrame(expanded_rows);
        print(f"generate_aggregated_data: Expanded to {len(expanded_annotations_df)} rows. Aggregating...")
        grouped = expanded_annotations_df.groupby(['Stereotype_State', 'Category', 'Attribute'])
        aggregated_data = grouped.agg(
            Stereotype_Votes=('annotation', lambda x: (x == 'Stereotype').sum()), Not_Stereotype_Votes=('annotation', lambda x: (x == 'Not a Stereotype').sum()),
            Not_Sure_Votes=('annotation', lambda x: (x == 'Not sure').sum()), Average_Offensiveness=('offensiveness_rating', calculate_mean_offensiveness)).reset_index()
        aggregated_data['Average_Offensiveness'] = aggregated_data['Average_Offensiveness'].round(2)
        print(f"generate_aggregated_data: Aggregation complete. Result has {len(aggregated_data)} rows.")
        aggregated_df = aggregated_data
    # --- END OF UNCHANGED PART ---

    except (MySQLError, pd.errors.DatabaseError) as e:
        print(f"ERROR generate_aggregated_data: Database error: {e}")
        flash(f"Error generating aggregated data: Database issue.", "error")
        aggregated_df = None # Indicate failure
    except ValueError as e: # Catch validation errors
        print(f"ERROR generate_aggregated_data: Value error during processing: {e}")
        flash(f"Error generating aggregated data: {e}", "error")
        aggregated_df = None
    except Exception as e:
        print(f"UNEXPECTED ERROR generate_aggregated_data: {e}\n{traceback.format_exc()}")
        flash(f"Error generating aggregated data: An unexpected error occurred.", "error")
        aggregated_df = None
    finally:
        if db_conn_proc and db_conn_proc.is_connected():
            try: db_conn_proc.close()
            except Exception as e_close: print(f"Warning: Error closing processing connection: {e_close}")
        print("--- generate_aggregated_data: Finished ---")

    return aggregated_df # Return DataFrame or None


# --- Flask Routes ---
# (Routes index, quiz, submit, thank_you, admin_view, download_processed_data, download_raw_data
#  remain functionally the same, they just rely on the hardcoded config via app.config)

@app.route('/', methods=['GET', 'POST'])
def index():
    db_status_ok = app.config.get('DB_CONFIGURED', False)
    if not db_status_ok:
         # This warning should NOT appear if hardcoding is correct
         flash("Warning: Database is not configured (Hardcoded issue?). Data submission will not work.", "warning")

    form_data = {}
    if request.method == 'POST':
        # --- POST logic remains the same ---
        form_data = request.form
        user_name = request.form.get('name', '').strip(); user_state = request.form.get('user_state')
        user_age = request.form.get('age','').strip(); user_sex = request.form.get('sex','')
        errors = False
        if not user_name: flash('Name is required.', 'error'); errors = True
        valid_states = [s for s in INDIAN_STATES if not s.startswith("Error:")]
        if not user_state or user_state not in valid_states: flash('Please select a valid state.', 'error'); errors = True
        if user_age:
            try: age_val = int(user_age); assert 0 <= age_val <= 130
            except: flash('Age must be a valid number.', 'error'); errors = True
        if errors: return render_template('index.html', states=INDIAN_STATES, form_data=form_data, db_status_ok=db_status_ok)
        user_info = {'name': user_name, 'state': user_state, 'age': user_age or None, 'sex': user_sex or None}
        return redirect(url_for('quiz', **user_info))
        # --- End of POST logic ---

    return render_template('index.html', states=INDIAN_STATES, form_data=form_data, db_status_ok=db_status_ok)

@app.route('/quiz')
def quiz():
    # --- Quiz logic remains the same ---
    user_info = {'name': request.args.get('name'),'state': request.args.get('state'),'age': request.args.get('age'),'sex': request.args.get('sex')}
    if not user_info['name'] or not user_info['state']:
         flash('User info missing. Please start again.', 'error'); return redirect(url_for('index'))
    if not ALL_STEREOTYPE_DATA or (INDIAN_STATES and INDIAN_STATES[0].startswith("Error:")):
        flash('Error: Stereotype data not loaded.', 'error'); return redirect(url_for('index'))
    target_state = user_info['state']
    filtered_quiz_items = [item for item in ALL_STEREOTYPE_DATA if item['state'] == target_state]
    if not filtered_quiz_items: flash(f"No items found for {target_state}.", 'info')
    return render_template('quiz.html', quiz_items=filtered_quiz_items, user_info=user_info)
    # --- End of Quiz logic ---


@app.route('/submit', methods=['POST'])
def submit():
    # --- Submit logic remains the same, relying on get_db() which uses hardcoded config ---
    if not app.config.get('DB_CONFIGURED', False):
        print("ERROR submit: Attempted submission, but DB is not configured (Hardcoded issue?).")
        flash("Submission failed: Database is not configured.", 'error')
        return redirect(url_for('index'))
    cursor = get_db()
    if not cursor: return redirect(url_for('index'))
    db_connection = getattr(g, 'db', None)
    if not db_connection or not db_connection.is_connected():
        print("ERROR submit: DB connection object not found or closed."); flash("Internal server error.", "error"); return redirect(url_for('index'))

    try:
        user_name = request.form.get('user_name'); user_state = request.form.get('user_state')
        user_age_str = request.form.get('user_age'); user_sex = request.form.get('user_sex') or None
        if not user_name or not user_state: flash("User info missing.", 'error'); return redirect(url_for('index'))
        user_age = None
        if user_age_str and user_age_str.strip():
            try: user_age = int(user_age_str.strip()); assert 0 <= user_age <= 130
            except: flash("Invalid age format.", "warning"); user_age = None

        results_to_insert = []; processed_indices = set()
        for key in request.form:
             if key.startswith('annotation_'):
                 try:
                     parts = key.split('_'); identifier = parts[-1]
                     if not identifier.isdigit() or identifier in processed_indices: continue
                     processed_indices.add(identifier)
                     category = request.form.get(f'category_{identifier}'); superset = request.form.get(f'superset_{identifier}'); annotation = request.form.get(key)
                     if not all([category, superset, annotation]): print(f"Warning submit: Missing data for item {identifier}. Skip."); continue
                     offensiveness = -1
                     if annotation == 'Stereotype':
                         rating_str = request.form.get(f'offensiveness_{identifier}')
                         if rating_str is not None and rating_str.isdigit():
                             try: rating_val = int(rating_str); offensiveness = rating_val if 0 <= rating_val <= 5 else -1
                             except ValueError: pass # Keep -1
                     results_to_insert.append({'user_name': user_name, 'user_state': user_state, 'user_age': user_age, 'user_sex': user_sex, 'category': category, 'attribute_superset': superset, 'annotation': annotation, 'offensiveness_rating': offensiveness})
                 except Exception as item_err: print(f"Error processing form item key '{key}': {item_err}")

        if results_to_insert:
             sql = f"""INSERT INTO {RESULTS_TABLE} (user_name, user_state, user_age, user_sex, category, attribute_superset, annotation, offensiveness_rating) VALUES (%(user_name)s, %(user_state)s, %(user_age)s, %(user_sex)s, %(category)s, %(attribute_superset)s, %(annotation)s, %(offensiveness_rating)s)"""
             try:
                 cursor.executemany(sql, results_to_insert); db_connection.commit()
                 flash(f"Thank you! Your {len(results_to_insert)} responses recorded.", 'success')
             except MySQLError as db_err:
                  print(f"CRITICAL submit DB INSERT ERROR: {db_err}"); print(f"Data: {results_to_insert[:2]}...")
                  try: db_connection.rollback()
                  except Exception as rb_err: print(f"Rollback failed: {rb_err}")
                  flash("Database error saving responses.", 'error'); return redirect(url_for('index'))
             except Exception as e:
                  print(f"CRITICAL submit UNEXPECTED INSERT ERROR: {e}\n{traceback.format_exc()}")
                  try: db_connection.rollback()
                  except Exception as rb_err: print(f"Rollback failed: {rb_err}")
                  flash("Unexpected error saving data.", 'error'); return redirect(url_for('index'))
        else:
             flash("No valid responses processed.", 'warning'); return redirect(url_for('index')) # Or quiz?
        return redirect(url_for('thank_you'))
    except Exception as e:
        print(f"CRITICAL submit ROUTE UNEXPECTED ERROR: {e}\n{traceback.format_exc()}"); flash("Unexpected error during submission.", 'error')
        db_conn = getattr(g, 'db', None)
        if db_conn and db_conn.is_connected():
            try: db_conn.rollback()
            except Exception as rb_err: print(f"Rollback attempt failed in outer catch: {rb_err}")
        return redirect(url_for('index'))
    # --- End of Submit logic ---


@app.route('/thank_you')
def thank_you():
    # --- Thank you logic remains the same ---
    return render_template('thank_you.html')

@app.route('/admin')
def admin_view():
    # --- Admin view logic remains the same ---
    db_status_ok=app.config.get('DB_CONFIGURED', False)
    if not db_status_ok: flash("DB not configured.", "error"); return render_template('admin.html', results=[], db_status_ok=False)
    cursor = get_db()
    if not cursor: return render_template('admin.html', results=[], db_status_ok=False)
    results_data = []
    try:
        cursor.execute(f"SELECT * FROM {RESULTS_TABLE} ORDER BY timestamp DESC"); results_data = cursor.fetchall()
    except Exception as err: print(f"Admin View Error: {err}"); flash(f'Error fetching results: {err}', 'error')
    return render_template('admin.html', results=results_data, db_status_ok=True)

@app.route('/admin/download_processed')
def download_processed_data():
    # --- Download processed logic remains the same ---
    aggregated_df = generate_aggregated_data()
    if aggregated_df is None: return redirect(url_for('admin_view'))
    if aggregated_df.empty: flash("No data to process.", "warning"); return redirect(url_for('admin_view'))
    try:
        buffer = io.BytesIO(); aggregated_df.to_csv(buffer, index=False, encoding='utf-8'); buffer.seek(0)
        return send_file(buffer, mimetype='text/csv', download_name='final_aggregated_stereotypes.csv', as_attachment=True)
    except Exception as e: print(f"Download Processed Error: {e}"); flash(f"Error creating file: {e}", "error"); return redirect(url_for('admin_view'))

@app.route('/admin/download_raw')
def download_raw_data():
    # --- Download raw logic remains the same ---
    if not app.config.get('DB_CONFIGURED'): flash("DB not configured.", "error"); return redirect(url_for('admin_view'))
    db_conn_raw = None
    try:
        db_conn_raw = mysql.connector.connect(host=app.config['MYSQL_HOST'], user=app.config['MYSQL_USER'], password=app.config['MYSQL_PASSWORD'], database=app.config['MYSQL_DB'], port=app.config.get('MYSQL_PORT', 3306), connection_timeout=15)
        if not db_conn_raw.is_connected(): raise MySQLError("Raw Download: Failed connection.")
        raw_results_df = pd.read_sql_query(f"SELECT * FROM {RESULTS_TABLE} ORDER BY timestamp DESC", db_conn_raw)
        if raw_results_df.empty: flash("No raw results found.", "warning"); return redirect(url_for('admin_view'))
        buffer = io.BytesIO(); raw_results_df.to_csv(buffer, index=False, encoding='utf-8'); buffer.seek(0)
        return send_file(buffer, mimetype='text/csv', download_name='raw_quiz_results.csv', as_attachment=True)
    except Exception as e: print(f"ERROR Download Raw: {e}\n{traceback.format_exc()}"); flash(f"Error fetching/preparing raw data: {e}", "error"); return redirect(url_for('admin_view'))
    finally:
        if db_conn_raw and db_conn_raw.is_connected():
            try: db_conn_raw.close()
            except Exception as e_close: print(f"Warning: Error closing raw download connection: {e_close}")

# Remove the __main__ block for deployment
# if __name__ == '__main__':
#    app.run(debug=True)