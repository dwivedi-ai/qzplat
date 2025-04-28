# stereotype_quiz_app/app.py

import os
import csv
import io
import mysql.connector
from mysql.connector import Error as MySQLError
# Ensure get_flashed_messages is imported
from flask import (Flask, render_template, request, redirect, url_for, g,
                   flash, Response, send_file, get_flashed_messages)
import pandas as pd
import numpy as np
import traceback
# We don't need urlparse if reading individual components
# from urllib.parse import urlparse

# --- Configuration ---
CSV_FILE_PATH = os.path.join('data', 'stereotypes.csv')
SCHEMA_FILE = 'schema_mysql.sql'

# --- Configuration from Environment Variables ---
# Reading variables EXACTLY as Railway injects them (no underscores usually)

SECRET_KEY = os.environ.get('SECRET_KEY', 'a_very_weak_default_secret_key_change_me_if_local')

# Read individual components directly from Railway's injected variables
MYSQL_HOST = os.environ.get('MYSQLHOST')       # Read MYSQLHOST
MYSQL_USER = os.environ.get('MYSQLUSER')       # Read MYSQLUSER
MYSQL_PASSWORD = os.environ.get('MYSQLPASSWORD') # Read MYSQLPASSWORD
MYSQL_DB = os.environ.get('MYSQLDATABASE')     # Read MYSQLDATABASE

# Read MYSQLPORT as string and convert safely
db_port_str = os.environ.get('MYSQLPORT')
MYSQL_PORT = None # Default to None if not set or invalid
if db_port_str:
    try:
        MYSQL_PORT = int(db_port_str)
    except ValueError:
        print(f"CRITICAL ERROR: Invalid value received for MYSQLPORT: '{db_port_str}'. Database connection will likely fail.")
else:
    print("CRITICAL ERROR: MYSQLPORT environment variable not found.")


RESULTS_TABLE = 'results'

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY

# Populate app.config ONLY if essential components were found
if all([MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB, MYSQL_PORT is not None]):
    app.config['MYSQL_HOST'] = MYSQL_HOST
    app.config['MYSQL_USER'] = MYSQL_USER
    app.config['MYSQL_PASSWORD'] = MYSQL_PASSWORD
    app.config['MYSQL_DB'] = MYSQL_DB
    app.config['MYSQL_PORT'] = MYSQL_PORT # This will be an int if successful

    # Print config being used (helps debugging)
    print("--- Application Configuration (Reading Direct Railway Vars) ---")
    print(f"SECRET_KEY: {'Set' if SECRET_KEY != 'a_very_weak_default_secret_key_change_me_if_local' else 'Default (UNSAFE!)'}")
    print(f"MYSQL_HOST: {app.config.get('MYSQL_HOST')} (from MYSQLHOST env var)")
    print(f"MYSQL_USER: {app.config.get('MYSQL_USER')} (from MYSQLUSER env var)")
    print(f"MYSQL_PASSWORD: {'Set' if app.config.get('MYSQL_PASSWORD') else 'Not Set'} (from MYSQLPASSWORD env var)")
    print(f"MYSQL_DB: {app.config.get('MYSQL_DB')} (from MYSQLDATABASE env var)")
    print(f"MYSQL_PORT: {app.config.get('MYSQL_PORT')} (from MYSQLPORT env var)")
    print("-----------------------------------------------------------")
else:
    print("--- Application Configuration FAILED: Missing essential DB environment variables ---")
    # Set a flag or handle this state if needed, otherwise get_db/init_db will fail
    app.config['DB_CONFIGURED'] = False


# --- Database Functions ---
def get_db():
    """Opens a new MySQL database connection and cursor if none exist for the current request context."""
    if 'db' not in g:
        # Check if DB config is available in app.config first
        if not app.config.get('DB_CONFIGURED', True): # Check if config failed earlier
             print("ERROR get_db: Database configuration failed during startup.")
             flash('Database configuration error. Please contact admin.', 'error')
             g.db = None; g.cursor = None
             return None
        # Check again specifically for needed keys (belt and suspenders)
        if not all(k in app.config for k in ['MYSQL_HOST', 'MYSQL_USER', 'MYSQL_PASSWORD', 'MYSQL_DB', 'MYSQL_PORT']):
             print("ERROR get_db: Database configuration missing in app.config.")
             flash('Database configuration error. Please contact admin.', 'error')
             g.db = None; g.cursor = None
             return None

        try:
            g.db = mysql.connector.connect(
                host=app.config['MYSQL_HOST'], user=app.config['MYSQL_USER'],
                password=app.config['MYSQL_PASSWORD'], database=app.config['MYSQL_DB'],
                port=app.config['MYSQL_PORT'], connection_timeout=10
            )
            g.cursor = g.db.cursor(dictionary=True)
        except MySQLError as err:
            print(f"ERROR connecting to MySQL: {err}")
            print(f"DEBUG: Connection attempted with Host={app.config.get('MYSQL_HOST')}, User={app.config.get('MYSQL_USER')}, DB={app.config.get('MYSQL_DB')}, Port={app.config.get('MYSQL_PORT')}")
            flash('Database connection error. Please try again later or contact admin.', 'error')
            g.db = None; g.cursor = None
    return getattr(g, 'cursor', None)

@app.teardown_appcontext
def close_db(error):
    # (Keep close_db as before)
    cursor = g.pop('cursor', None); db = g.pop('db', None)
    if cursor:
        try: cursor.close()
        except Exception as e: print(f"Error closing cursor: {e}")
    if db and db.is_connected():
        try: db.close()
        except Exception as e: print(f"Error closing DB connection: {e}")
    if error: print(f"App context teardown error detected: {error}")


def init_db():
    """Connects to MySQL, creates the database IF NOT EXISTS, and creates the table IF NOT EXISTS using schema_mysql.sql."""
    # Check if DB config is available
    if not app.config.get('DB_CONFIGURED', True) or not all(k in app.config for k in ['MYSQL_HOST', 'MYSQL_USER', 'MYSQL_PASSWORD', 'MYSQL_DB', 'MYSQL_PORT']):
         print("CRITICAL init_db ERROR: Database configuration missing. Skipping initialization.")
         return

    # (Rest of init_db logic remains the same as it reads from app.config)
    temp_conn = None; temp_cursor = None
    db_host = app.config['MYSQL_HOST']; db_user = app.config['MYSQL_USER']
    db_password = app.config['MYSQL_PASSWORD']; db_port = app.config['MYSQL_PORT']
    db_name = app.config['MYSQL_DB']
    print(f"--- init_db: Attempting DB Initialization (Host: {db_host}, User: {db_user}, DB: {db_name}, Port: {db_port}) ---")
    try:
        print(f"init_db: Connecting to MySQL server ({db_host}:{db_port}) to check/create database...")
        temp_conn = mysql.connector.connect(
            host=db_host, user=db_user, password=db_password, port=db_port, connection_timeout=15
        )
        temp_cursor = temp_conn.cursor()
        print(f"init_db: Connected to MySQL server. Checking/creating database '{db_name}'...")
        try:
            temp_cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            temp_conn.commit(); print(f"init_db: Database '{db_name}' checked/created.")
        except MySQLError as err:
            print(f"Warning init_db: Could not CREATE DATABASE (might be normal/permissions issue): {err}.")
        try:
            temp_cursor.execute(f"USE `{db_name}`"); print(f"init_db: Switched to database '{db_name}'.")
        except MySQLError as err: print(f"CRITICAL init_db ERROR: Failed switch to DB '{db_name}': {err}."); return
        print(f"init_db: Checking for table '{RESULTS_TABLE}'..."); temp_cursor.execute(f"SHOW TABLES LIKE '{RESULTS_TABLE}'")
        table_exists = temp_cursor.fetchone()
        if not table_exists:
            print(f"init_db: Table '{RESULTS_TABLE}' not found. Creating..."); schema_path = os.path.join(app.root_path, SCHEMA_FILE)
            if not os.path.exists(schema_path):
                schema_path_alt = os.path.join(app.root_path, 'schema_mysql.sql')
                if os.path.exists(schema_path_alt): schema_path = schema_path_alt
                else: raise FileNotFoundError(f"init_db: Schema file not found: {schema_path} or {schema_path_alt}")
            try:
                with open(schema_path, mode='r', encoding='utf-8') as f: sql_script = f.read()
                print(f"init_db: Executing SQL script from {schema_path}...")
                statement_results = temp_cursor.execute(sql_script, multi=True)
                for i, result in enumerate(statement_results):
                    if result.with_rows: result.fetchall() # Consume results
                temp_conn.commit(); print(f"init_db: Database table '{RESULTS_TABLE}' created.")
            except FileNotFoundError as e: print(f"CRITICAL init_db ERROR: Schema file missing: {e}.")
            except MySQLError as e: print(f"CRITICAL init_db ERROR executing schema: {e}"); temp_conn.rollback()
            except Exception as e: print(f"CRITICAL init_db UNEXPECTED ERROR schema: {e}\n{traceback.format_exc()}"); temp_conn.rollback()
        else: print(f"init_db: Database table '{RESULTS_TABLE}' exists.")
    except MySQLError as e: print(f"CRITICAL init_db ERROR connection/setup: {e}")
    except Exception as e: print(f"CRITICAL init_db UNEXPECTED error: {e}\n{traceback.format_exc()}")
    finally:
        if temp_cursor: temp_cursor.close()
        if temp_conn and temp_conn.is_connected(): temp_conn.close()
        print("--- init_db: Finished DB Initialization Check ---")

# --- Initialize DB on Application Start ---
# Only run init_db if database was configured successfully
if app.config.get('MYSQL_HOST'): # Check if host was set (implies successful reading)
    print(">>> Application starting: Performing database initialization check...")
    with app.app_context(): init_db()
    print(">>> Application starting: Database initialization check complete.")
else:
    print(">>> Application starting: Skipping DB initialization due to configuration error.")


# --- Data Loading Function ---
# (Keep load_stereotype_data exactly as before)
def load_stereotype_data(relative_filepath=CSV_FILE_PATH):
    stereotype_data = []; full_filepath = os.path.join(app.root_path, relative_filepath)
    # print(f"--- load_stereotype_data: Loading from: {full_filepath} ---") # Less verbose
    try:
        if not os.path.exists(full_filepath): raise FileNotFoundError(f"Not found: {full_filepath}")
        with open(full_filepath, mode='r', encoding='utf-8-sig') as infile:
            reader = csv.DictReader(infile); required_cols = ['State', 'Category', 'Superset', 'Subsets']
            if not reader.fieldnames or not all(f in reader.fieldnames for f in required_cols):
                 raise ValueError(f"CSV missing required cols. Found: {reader.fieldnames}")
            row_count, error_count = 0, 0
            for i, row in enumerate(reader):
                row_count += 1; state = row.get('State','').strip(); category = row.get('Category','').strip() or 'Uncategorized'
                superset = row.get('Superset','').strip(); subsets_str = row.get('Subsets','')
                try:
                    if not state or not superset: error_count += 1; continue
                    subsets = sorted([s.strip() for s in subsets_str.split(',') if s.strip()])
                    stereotype_data.append({'state': state,'category': category,'superset': superset,'subsets': subsets})
                except Exception as row_err: print(f"Err row {i+2}: {row_err}"); error_count += 1; continue
        # if not stereotype_data and row_count > 0: print(f"Warn: Loaded 0 entries from {row_count} rows.") # Less verbose
        # elif error_count > 0: print(f"Loaded {len(stereotype_data)} entries ({error_count} rows skipped).")
        # else: print(f"Loaded {len(stereotype_data)} entries.")
        return stereotype_data
    except FileNotFoundError as e: print(f"FATAL: Stereotype file not found: {e}"); return []
    except ValueError as e: print(f"FATAL: CSV format error: {e}"); return []
    except Exception as e: print(f"FATAL loading stereotypes: {e}\n{traceback.format_exc()}"); return []
    # finally: print("--- load_stereotype_data: Finished ---") # Less verbose


# --- Load Data & States ---
print(">>> Loading stereotype definitions...")
ALL_STEREOTYPE_DATA = load_stereotype_data()
INDIAN_STATES = sorted(list(set(item['state'] for item in ALL_STEREOTYPE_DATA))) if ALL_STEREOTYPE_DATA else []
if not ALL_STEREOTYPE_DATA or not INDIAN_STATES:
    print("\nCRITICAL WARNING: Stereotype data loading failed/empty.\n")
    INDIAN_STATES = ["Error: State data unavailable"]
else: print(f">>> States available: {len(INDIAN_STATES)}")


# --- Data Processing Logic ---
# (Keep calculate_mean_offensiveness and generate_aggregated_data exactly as before)
def calculate_mean_offensiveness(series):
    valid_ratings = series[series >= 0]; return valid_ratings.mean() if not valid_ratings.empty else np.nan
def generate_aggregated_data():
    # print("--- [Processing] Starting data aggregation ---") # Less verbose
    db_conn_proc = None; aggregated_df = None
    try:
        if not all(k in app.config for k in ['MYSQL_HOST', 'MYSQL_USER', 'MYSQL_PASSWORD', 'MYSQL_DB', 'MYSQL_PORT']):
            raise ValueError("Processing Error: Database configuration not available.")
        # print("[Processing] Connecting to DB...") # Less verbose
        db_conn_proc = mysql.connector.connect(
            host=app.config['MYSQL_HOST'], user=app.config['MYSQL_USER'], password=app.config['MYSQL_PASSWORD'],
            database=app.config['MYSQL_DB'], port=app.config['MYSQL_PORT'], connection_timeout=10)
        if not db_conn_proc.is_connected(): raise MySQLError("Processing: Failed connection.")
        results_df = pd.read_sql_query(f"SELECT * FROM {RESULTS_TABLE}", db_conn_proc)
        # print(f"[Processing] Loaded {len(results_df)} results."); # Less verbose
        if results_df.empty: return pd.DataFrame()
        results_df['Stereotype_State'] = results_df['user_state']
        stereotypes_path = os.path.join(app.root_path, CSV_FILE_PATH)
        if not os.path.exists(stereotypes_path): raise FileNotFoundError(f"Processing: Defs not found: {stereotypes_path}")
        stereotypes_df = pd.read_csv(stereotypes_path, encoding='utf-8-sig')
        required_def_cols = ['State', 'Category', 'Superset', 'Subsets']
        if not all(col in stereotypes_df.columns for col in required_def_cols): raise ValueError("Processing: Defs CSV missing cols.")
        stereotypes_df['Subsets_List'] = stereotypes_df['Subsets'].fillna('').astype(str).apply(lambda x: sorted([s.strip() for s in x.split(',') if s.strip()]))
        subset_lookup = stereotypes_df.set_index(['State', 'Category', 'Superset'])['Subsets_List'].to_dict()
        expanded_rows = []; processing_errors = 0
        for index, result_row in results_df.iterrows():
            state = result_row.get('Stereotype_State'); category = result_row.get('category'); superset = result_row.get('attribute_superset')
            annotation = result_row.get('annotation'); rating_val = result_row.get('offensiveness_rating'); rating = int(rating_val) if pd.notna(rating_val) else -1
            if not all([state, category, superset, annotation]): processing_errors += 1; continue
            expanded_rows.append({'Stereotype_State': state, 'Category': category, 'Attribute': superset, 'annotation': annotation, 'offensiveness_rating': rating})
            subsets_list = subset_lookup.get((state, category, superset), [])
            for subset in subsets_list: expanded_rows.append({'Stereotype_State': state, 'Category': category, 'Attribute': subset, 'annotation': annotation, 'offensiveness_rating': rating})
        # if processing_errors > 0: print(f"[Processing] Note: Skipped {processing_errors} rows.") # Less verbose
        if not expanded_rows: return pd.DataFrame()
        expanded_annotations_df = pd.DataFrame(expanded_rows);
        grouped = expanded_annotations_df.groupby(['Stereotype_State', 'Category', 'Attribute'])
        aggregated_data = grouped.agg(
            Stereotype_Votes=('annotation', lambda x: (x == 'Stereotype').sum()), Not_Stereotype_Votes=('annotation', lambda x: (x == 'Not a Stereotype').sum()),
            Not_Sure_Votes=('annotation', lambda x: (x == 'Not sure').sum()), Average_Offensiveness=('offensiveness_rating', calculate_mean_offensiveness)).reset_index()
        aggregated_data['Average_Offensiveness'] = aggregated_data['Average_Offensiveness'].round(2)
        # print(f"[Processing] Aggregation complete ({len(aggregated_data)} rows)."); # Less verbose
        aggregated_df = aggregated_data
    except FileNotFoundError as e: print(f"ERROR [Proc]: File not found: {e}"); flash(f"Error: Data file not found.", "error"); aggregated_df = None
    except (MySQLError, pd.errors.DatabaseError) as e: print(f"ERROR [Proc]: DB error: {e}"); flash(f"Error: Database issue.", "error"); aggregated_df = None
    except KeyError as e: print(f"ERROR [Proc]: Missing col: {e}"); flash(f"Error: Data mismatch.", "error"); aggregated_df = None
    except ValueError as e: print(f"ERROR [Proc]: {e}"); flash(f"Error: {e}", "error"); aggregated_df = None
    except Exception as e: print(f"UNEXPECTED ERROR [Proc]:\n{traceback.format_exc()}"); flash(f"Error: Unexpected processing error.", "error"); aggregated_df = None
    finally:
        if db_conn_proc and db_conn_proc.is_connected(): db_conn_proc.close()
    return aggregated_df


# --- Flask Routes ---
# (Keep index, quiz, submit, thank_you, admin routes exactly as before)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_name = request.form.get('name', '').strip(); user_state = request.form.get('user_state')
        user_age = request.form.get('age','').strip(); user_sex = request.form.get('sex','')
        errors = False
        if not user_name: flash('Name is required.', 'error'); errors = True
        valid_states = [s for s in INDIAN_STATES if not s.startswith("Error:")]
        if not user_state or user_state not in valid_states: flash('Please select a valid state.', 'error'); errors = True
        if user_age:
            try: age_val = int(user_age); assert age_val >= 0
            except: flash('Age must be a valid number.', 'error'); errors = True
        if errors: return render_template('index.html', states=INDIAN_STATES, form_data=request.form)
        user_info = {'name': user_name, 'state': user_state, 'age': user_age, 'sex': user_sex}
        return redirect(url_for('quiz', **user_info))
    # Check if DB configured before rendering potentially problematic form
    if not app.config.get('DB_CONFIGURED', True):
         flash("Application database is not configured. Please contact administrator.", "error")
         # Render a simple error page or just the message?
         # return "Database configuration error", 500
    return render_template('index.html', states=INDIAN_STATES, form_data={})

@app.route('/quiz')
def quiz():
    user_info = {'name': request.args.get('name'),'state': request.args.get('state'),'age': request.args.get('age'),'sex': request.args.get('sex')}
    if not user_info['name'] or not user_info['state']:
         flash('User info missing. Please start again.', 'error'); return redirect(url_for('index'))
    if not ALL_STEREOTYPE_DATA or (INDIAN_STATES and INDIAN_STATES[0].startswith("Error:")):
        flash('Error: Stereotype data not loaded.', 'error'); return redirect(url_for('index'))
    target_state = user_info['state']
    filtered_quiz_items = [item for item in ALL_STEREOTYPE_DATA if item['state'] == target_state]
    if not filtered_quiz_items: flash(f"No items found for {target_state}.", 'info')
    return render_template('quiz.html', quiz_items=filtered_quiz_items, user_info=user_info)

@app.route('/submit', methods=['POST'])
def submit():
    cursor = get_db()
    if not cursor: # get_db handles flashing error if config/connection fails
        print("Submit Error: DB connection failed.")
        return redirect(url_for('index'))
    db_connection = getattr(g, 'db', None)
    try:
        user_name = request.form.get('user_name'); user_state = request.form.get('user_state')
        user_age_str = request.form.get('user_age'); user_sex = request.form.get('user_sex') or None
        if not user_name or not user_state:
            flash("User info missing.", 'error'); return redirect(url_for('index'))
        user_age = None
        if user_age_str:
            try: user_age = int(user_age_str); assert user_age >= 0
            except: flash("Invalid age.", "warning"); user_age = None
        results_to_insert = []; processed_indices = set()
        for key in request.form:
             if key.startswith('annotation_'):
                 identifier = key.split('_')[-1]
                 try: assert identifier.isdigit(); assert identifier not in processed_indices
                 except: continue
                 processed_indices.add(identifier); superset = request.form.get(f'superset_{identifier}')
                 category = request.form.get(f'category_{identifier}'); annotation = request.form.get(key)
                 if not all([superset, category, annotation]): continue
                 offensiveness = -1
                 if annotation == 'Stereotype':
                     rating_str = request.form.get(f'offensiveness_{identifier}')
                     if rating_str is not None and rating_str != '':
                         try: offensiveness = int(rating_str); assert 0 <= offensiveness <= 5
                         except: offensiveness = -1
                 results_to_insert.append({'user_name': user_name,'user_state': user_state,'user_age': user_age,'user_sex': user_sex,
                                           'category': category,'attribute_superset': superset,'annotation': annotation,'offensiveness_rating': offensiveness})
        if results_to_insert:
             sql = f"INSERT INTO {RESULTS_TABLE} (user_name, user_state, user_age, user_sex, category, attribute_superset, annotation, offensiveness_rating) VALUES (%(user_name)s, %(user_state)s, %(user_age)s, %(user_sex)s, %(category)s, %(attribute_superset)s, %(annotation)s, %(offensiveness_rating)s)"
             try:
                 cursor.executemany(sql, results_to_insert)
                 if db_connection: db_connection.commit(); flash(f"Submitted {len(results_to_insert)} responses. Thank you!", 'success')
                 else: flash("Internal error.", "error"); return redirect(url_for('index')) # Should not happen
             except MySQLError as db_err:
                  print(f"DB INSERT ERROR: {db_err}")
                  try:
                      if db_connection: db_connection.rollback()
                  except: pass
                  flash("DB error saving responses.", 'error'); return redirect(url_for('index'))
             except Exception as e:
                  print(f"UNEXPECTED INSERT ERROR: {e}\n{traceback.format_exc()}")
                  try:
                      if db_connection: db_connection.rollback()
                  except: pass
                  flash("Unexpected error saving data.", 'error'); return redirect(url_for('index'))
        else: flash("No valid responses found.", 'warning')
        return redirect(url_for('thank_you'))
    except Exception as e:
        print(f"SUBMIT ROUTE UNEXPECTED ERROR: {e}\n{traceback.format_exc()}"); flash("Unexpected submission error.", 'error')
        try:
            db_conn = getattr(g, 'db', None)
            if db_conn and db_conn.is_connected(): db_conn.rollback()
        except: pass
        return redirect(url_for('index'))

@app.route('/thank_you')
def thank_you():
    return render_template('thank_you.html')

@app.route('/admin')
def admin_view():
    cursor = get_db()
    if not cursor: flash("DB connection failed.", "error"); return redirect(url_for('index'))
    results_data = []
    try:
        query = f"SELECT * FROM {RESULTS_TABLE} ORDER BY timestamp DESC"
        cursor.execute(query)
        results_data = cursor.fetchall()
    except MySQLError as err:
        print(f"Admin DB Error: {err}")
        flash(f'Error fetching results: {err}', 'error')
    except Exception as e: print(f"Admin Unexpected Error: {e}\n{traceback.format_exc()}"); flash('Unexpected error.', 'error')
    return render_template('admin.html', results=results_data)

@app.route('/admin/download_processed')
def download_processed_data():
    aggregated_df = generate_aggregated_data()
    if aggregated_df is None: return redirect(url_for('admin_view'))
    if aggregated_df.empty: flash("No data to process.", "warning"); return redirect(url_for('admin_view'))
    try:
        buffer = io.BytesIO(); aggregated_df.to_csv(buffer, index=False, encoding='utf-8'); buffer.seek(0)
        name = 'final_aggregated_stereotypes.csv'; return send_file(buffer, mimetype='text/csv', download_name=name, as_attachment=True)
    except Exception as e: print(f"Download Proc Err: {e}\n{traceback.format_exc()}"); flash(f"Error creating file: {e}", "error"); return redirect(url_for('admin_view'))

@app.route('/admin/download_raw')
def download_raw_data():
    db_conn_raw = None
    try:
        if not all(k in app.config for k in ['MYSQL_HOST', 'MYSQL_USER', 'MYSQL_PASSWORD', 'MYSQL_DB', 'MYSQL_PORT']):
             raise ValueError("Raw Download Error: Database configuration not available.")
        db_conn_raw = mysql.connector.connect(host=app.config['MYSQL_HOST'], user=app.config['MYSQL_USER'], password=app.config['MYSQL_PASSWORD'], database=app.config['MYSQL_DB'], port=app.config['MYSQL_PORT'], connection_timeout=10 )
        if not db_conn_raw.is_connected(): raise MySQLError("Raw DL: Failed connection.")
        query = f"SELECT * FROM {RESULTS_TABLE} ORDER BY timestamp DESC"; raw_results_df = pd.read_sql_query(query, db_conn_raw)
        if raw_results_df.empty: flash("Raw results empty.", "warning"); return redirect(url_for('admin_view'))
        buffer = io.BytesIO(); raw_results_df.to_csv(buffer, index=False, encoding='utf-8'); buffer.seek(0)
        name = 'raw_quiz_results.csv'; return send_file(buffer, mimetype='text/csv', download_name=name, as_attachment=True )
    except (MySQLError, pd.errors.DatabaseError, ValueError) as e: print(f"ERROR [Raw DL]: {e}"); flash(f"Error fetching raw data: {e}", "error"); return redirect(url_for('admin_view'))
    except Exception as e: print(f"UNEXPECTED ERROR [Raw DL]:\n{traceback.format_exc()}"); flash(f"Unexpected error preparing raw download: {e}", "error"); return redirect(url_for('admin_view'))
    finally:
        if db_conn_raw and db_conn_raw.is_connected(): db_conn_raw.close()

# Removed the __main__ block for deployment