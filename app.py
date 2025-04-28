# stereotype_quiz_app/app.py

import os
import csv
import io
import mysql.connector
from mysql.connector import Error as MySQLError
# FIX: Import get_flashed_messages
from flask import (Flask, render_template, request, redirect, url_for, g,
                   flash, Response, send_file, get_flashed_messages)
import pandas as pd
import numpy as np
import traceback

# --- Configuration ---
CSV_FILE_PATH = os.path.join('data', 'stereotypes.csv')
SCHEMA_FILE = 'schema_mysql.sql'

# --- Configuration from Environment Variables ---
# FIX: Looking for variables WITH underscores, matching your Railway setup.

SECRET_KEY = os.environ.get('SECRET_KEY', 'a_very_weak_default_secret_key_change_me_if_local')

# FIX: Expecting these variables (with underscores) to be set in Railway
MYSQL_HOST = os.environ.get('MYSQL_HOST', 'localhost') # Expect MYSQL_HOST in env
MYSQL_USER = os.environ.get('MYSQL_USER', 'stereotype_user') # Expect MYSQL_USER in env
MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD', 'RespAI@2025') # Expect MYSQL_PASSWORD in env
MYSQL_DB = os.environ.get('MYSQL_DB', 'stereotype_quiz_db') # Expect MYSQL_DB in env
MYSQL_PORT = int(os.environ.get('MYSQL_PORT', 3306)) # Expect MYSQL_PORT in env

RESULTS_TABLE = 'results'

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY

app.config['MYSQL_HOST'] = MYSQL_HOST
app.config['MYSQL_USER'] = MYSQL_USER
app.config['MYSQL_PASSWORD'] = MYSQL_PASSWORD
app.config['MYSQL_DB'] = MYSQL_DB
app.config['MYSQL_PORT'] = MYSQL_PORT

# Print config being used (helps debugging)
print("--- Application Configuration (Expecting Underscore Vars) ---")
print(f"SECRET_KEY: {'Set (likely from env)' if SECRET_KEY != 'a_very_weak_default_secret_key_change_me_if_local' else 'Using default (UNSAFE FOR PROD!)'}")
print(f"MYSQL_HOST: {app.config['MYSQL_HOST']} (read from MYSQL_HOST env var)")
print(f"MYSQL_USER: {app.config['MYSQL_USER']} (read from MYSQL_USER env var)")
print(f"MYSQL_PASSWORD: {'Set (likely from env)' if app.config['MYSQL_PASSWORD'] else 'Not Set'} (read from MYSQL_PASSWORD env var)")
print(f"MYSQL_DB: {app.config['MYSQL_DB']} (read from MYSQL_DB env var)")
print(f"MYSQL_PORT: {app.config['MYSQL_PORT']} (read from MYSQL_PORT env var)")
print("----------------------------------------------------------")

# --- Database Functions ---
# (Keep get_db, close_db, init_db exactly as they were - they read from app.config)
def get_db():
    """Opens a new MySQL database connection and cursor if none exist for the current request context."""
    if 'db' not in g:
        try:
            g.db = mysql.connector.connect(
                host=app.config['MYSQL_HOST'], user=app.config['MYSQL_USER'],
                password=app.config['MYSQL_PASSWORD'], database=app.config['MYSQL_DB'],
                port=app.config['MYSQL_PORT'], connection_timeout=10
            )
            g.cursor = g.db.cursor(dictionary=True)
        except MySQLError as err:
            print(f"ERROR connecting to MySQL: {err}")
            print(f"DEBUG: Connection attempted with Host={app.config['MYSQL_HOST']}, User={app.config['MYSQL_USER']}, DB={app.config['MYSQL_DB']}, Port={app.config['MYSQL_PORT']}")
            flash('Database connection error. Please try again later or contact admin.', 'error')
            g.db = None; g.cursor = None
    return getattr(g, 'cursor', None)

@app.teardown_appcontext
def close_db(error):
    """Closes the database cursor and connection at the end of the request."""
    cursor = g.pop('cursor', None)
    if cursor:
        try: cursor.close()
        except Exception as e: print(f"Error closing cursor: {e}")
    db = g.pop('db', None)
    if db and db.is_connected():
        try: db.close()
        except Exception as e: print(f"Error closing DB connection: {e}")
    # FIX: Check for NameError possibility during teardown as well
    if error and 'get_flashed_messages' not in str(error):
         print(f"App context teardown error detected: {error}")
    elif error:
         print(f"App context teardown error detected (pre-import fix?): {error}")


def init_db():
    """Connects to MySQL, creates the database IF NOT EXISTS, and creates the table IF NOT EXISTS using schema_mysql.sql."""
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
            temp_conn.commit()
            print(f"init_db: Database '{db_name}' checked/created.")
        except MySQLError as err:
            print(f"CRITICAL init_db ERROR: Failed to create database '{db_name}': {err}.")
            return
        try:
            temp_cursor.execute(f"USE `{db_name}`")
            print(f"init_db: Switched to database '{db_name}'.")
        except MySQLError as err:
            print(f"CRITICAL init_db ERROR: Failed to switch to database '{db_name}': {err}.")
            return
        print(f"init_db: Checking for table '{RESULTS_TABLE}' in database '{db_name}'...")
        temp_cursor.execute(f"SHOW TABLES LIKE '{RESULTS_TABLE}'")
        table_exists = temp_cursor.fetchone()
        if not table_exists:
            print(f"init_db: Table '{RESULTS_TABLE}' not found. Attempting creation...")
            schema_path = os.path.join(app.root_path, SCHEMA_FILE)
            if not os.path.exists(schema_path):
                schema_path_alt = os.path.join(app.root_path, 'schema_mysql.sql')
                if os.path.exists(schema_path_alt): schema_path = schema_path_alt
                else: raise FileNotFoundError(f"init_db: Schema file not found: {schema_path} or {schema_path_alt}")
            try:
                with open(schema_path, mode='r', encoding='utf-8') as f: sql_script = f.read()
                print(f"init_db: Executing SQL script from {schema_path}...")
                statement_results = temp_cursor.execute(sql_script, multi=True)
                for i, result in enumerate(statement_results):
                    print(f"  - Statement {i+1}: Rows: {result.rowcount if hasattr(result, 'rowcount') else 'N/A'}")
                    if result.with_rows: result.fetchall()
                temp_conn.commit(); print(f"init_db: Database table '{RESULTS_TABLE}' created.")
            except FileNotFoundError as fnf_err: print(f"CRITICAL init_db ERROR: Schema file not found: {fnf_err}.")
            except MySQLError as err: print(f"CRITICAL init_db ERROR executing schema: {err}"); temp_conn.rollback()
            except Exception as e: print(f"CRITICAL init_db UNEXPECTED ERROR: {e}\n{traceback.format_exc()}"); temp_conn.rollback()
        else: print(f"init_db: Database table '{RESULTS_TABLE}' already exists.")
    except MySQLError as err: print(f"CRITICAL init_db ERROR during connection/setup: {err}")
    except Exception as e: print(f"CRITICAL init_db UNEXPECTED error: {e}\n{traceback.format_exc()}")
    finally:
        if temp_cursor: temp_cursor.close()
        if temp_conn and temp_conn.is_connected(): temp_conn.close()
        print("--- init_db: Finished DB Initialization Check ---")


# --- Initialize DB on Application Start ---
print(">>> Application starting: Performing database initialization check...")
with app.app_context(): init_db()
print(">>> Application starting: Database initialization check complete.")

# --- Data Loading Function ---
# (Keep load_stereotype_data exactly as before)
def load_stereotype_data(relative_filepath=CSV_FILE_PATH):
    stereotype_data = []; full_filepath = os.path.join(app.root_path, relative_filepath)
    print(f"--- load_stereotype_data: Loading from: {full_filepath} ---")
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
        if not stereotype_data and row_count > 0: print(f"Warn: Loaded 0 entries from {row_count} rows.")
        elif error_count > 0: print(f"Loaded {len(stereotype_data)} entries ({error_count} rows skipped).")
        else: print(f"Loaded {len(stereotype_data)} entries.")
        return stereotype_data
    except FileNotFoundError as e: print(f"FATAL: Stereotype file not found: {e}"); return []
    except ValueError as e: print(f"FATAL: CSV format error: {e}"); return []
    except Exception as e: print(f"FATAL loading stereotypes: {e}\n{traceback.format_exc()}"); return []
    finally: print("--- load_stereotype_data: Finished ---")


# --- Load Data & States ---
print(">>> Loading stereotype definitions...")
ALL_STEREOTYPE_DATA = load_stereotype_data()
INDIAN_STATES = sorted(list(set(item['state'] for item in ALL_STEREOTYPE_DATA))) if ALL_STEREOTYPE_DATA else []
if not ALL_STEREOTYPE_DATA or not INDIAN_STATES:
    print("\nCRITICAL WARNING: Stereotype data loading failed/empty.\n")
    INDIAN_STATES = ["Error: State data unavailable"]
else: print(f">>> States available: {INDIAN_STATES}")

# --- Data Processing Logic ---
# (Keep calculate_mean_offensiveness and generate_aggregated_data exactly as before)
def calculate_mean_offensiveness(series):
    valid_ratings = series[series >= 0]; return valid_ratings.mean() if not valid_ratings.empty else np.nan
def generate_aggregated_data():
    print("--- [Processing] Starting data aggregation ---"); db_conn_proc = None; aggregated_df = None
    try:
        print("[Processing] Connecting to DB..."); db_conn_proc = mysql.connector.connect(
            host=app.config['MYSQL_HOST'], user=app.config['MYSQL_USER'], password=app.config['MYSQL_PASSWORD'],
            database=app.config['MYSQL_DB'], port=app.config['MYSQL_PORT'], connection_timeout=10)
        if not db_conn_proc.is_connected(): raise MySQLError("Processing: Failed connection.")
        print(f"[Processing] Fetching from {RESULTS_TABLE}..."); results_df = pd.read_sql_query(f"SELECT * FROM {RESULTS_TABLE}", db_conn_proc)
        print(f"[Processing] Loaded {len(results_df)} results.");
        if results_df.empty: print("[Processing] Table empty."); return pd.DataFrame()
        results_df['Stereotype_State'] = results_df['user_state']
        stereotypes_path = os.path.join(app.root_path, CSV_FILE_PATH)
        if not os.path.exists(stereotypes_path): raise FileNotFoundError(f"Processing: Defs not found: {stereotypes_path}")
        print(f"[Processing] Loading definitions: {stereotypes_path}"); stereotypes_df = pd.read_csv(stereotypes_path, encoding='utf-8-sig')
        required_def_cols = ['State', 'Category', 'Superset', 'Subsets']
        if not all(col in stereotypes_df.columns for col in required_def_cols): raise ValueError("Processing: Defs CSV missing cols.")
        stereotypes_df['Subsets_List'] = stereotypes_df['Subsets'].fillna('').astype(str).apply(lambda x: sorted([s.strip() for s in x.split(',') if s.strip()]))
        print(f"[Processing] Loaded {len(stereotypes_df)} definitions.")
        subset_lookup = stereotypes_df.set_index(['State', 'Category', 'Superset'])['Subsets_List'].to_dict()
        print("[Processing] Expanding annotations..."); expanded_rows = []; processing_errors = 0
        for index, result_row in results_df.iterrows():
            state = result_row.get('Stereotype_State'); category = result_row.get('category'); superset = result_row.get('attribute_superset')
            annotation = result_row.get('annotation'); rating_val = result_row.get('offensiveness_rating'); rating = int(rating_val) if pd.notna(rating_val) else -1
            if not all([state, category, superset, annotation]): processing_errors += 1; continue
            expanded_rows.append({'Stereotype_State': state, 'Category': category, 'Attribute': superset, 'annotation': annotation, 'offensiveness_rating': rating})
            subsets_list = subset_lookup.get((state, category, superset), [])
            for subset in subsets_list: expanded_rows.append({'Stereotype_State': state, 'Category': category, 'Attribute': subset, 'annotation': annotation, 'offensiveness_rating': rating})
        if processing_errors > 0: print(f"[Processing] Note: Skipped {processing_errors} rows.")
        if not expanded_rows: print("[Processing] No rows after expansion."); return pd.DataFrame()
        expanded_annotations_df = pd.DataFrame(expanded_rows); print(f"[Processing] Expanded to {len(expanded_annotations_df)} rows.")
        print("[Processing] Aggregating..."); grouped = expanded_annotations_df.groupby(['Stereotype_State', 'Category', 'Attribute'])
        aggregated_data = grouped.agg(
            Stereotype_Votes=('annotation', lambda x: (x == 'Stereotype').sum()), Not_Stereotype_Votes=('annotation', lambda x: (x == 'Not a Stereotype').sum()),
            Not_Sure_Votes=('annotation', lambda x: (x == 'Not sure').sum()), Average_Offensiveness=('offensiveness_rating', calculate_mean_offensiveness)).reset_index()
        aggregated_data['Average_Offensiveness'] = aggregated_data['Average_Offensiveness'].round(2)
        print(f"[Processing] Aggregation complete ({len(aggregated_data)} rows)."); print("--- [Processing] Finished successfully ---"); aggregated_df = aggregated_data
    except FileNotFoundError as e: print(f"ERROR [Proc]: File not found: {e}"); flash(f"Error: Data file not found.", "error"); aggregated_df = None
    except (MySQLError, pd.errors.DatabaseError) as e: print(f"ERROR [Proc]: DB error: {e}"); flash(f"Error: Database issue.", "error"); aggregated_df = None
    except KeyError as e: print(f"ERROR [Proc]: Missing col: {e}"); flash(f"Error: Data mismatch.", "error"); aggregated_df = None
    except ValueError as e: print(f"ERROR [Proc]: Value error: {e}"); flash(f"Error: Data format issue.", "error"); aggregated_df = None
    except Exception as e: print(f"UNEXPECTED ERROR [Proc]:\n{traceback.format_exc()}"); flash(f"Error: Unexpected processing error.", "error"); aggregated_df = None
    finally:
        if db_conn_proc and db_conn_proc.is_connected(): db_conn_proc.close(); print("[Processing] Closed DB connection.")
    return aggregated_df


# --- Flask Routes ---
# (Keep index and quiz routes exactly as before)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_name = request.form.get('name', '').strip(); user_state = request.form.get('user_state')
        user_age = request.form.get('age','').strip(); user_sex = request.form.get('sex','')
        errors = False
        if not user_name: flash('Name is required.', 'error'); errors = True
        valid_states = [s for s in INDIAN_STATES if not s.startswith("Error:")]
        if not user_state or user_state not in valid_states: flash('Please select a valid state.', 'error'); errors = True
        if user_age and not user_age.isdigit(): flash('Age must be a whole number.', 'error'); errors = True
        elif user_age and int(user_age) < 0: flash('Age cannot be negative.', 'error'); errors = True
        if errors: return render_template('index.html', states=INDIAN_STATES, form_data=request.form)
        user_info = {'name': user_name, 'state': user_state, 'age': user_age, 'sex': user_sex}
        print(f"Index POST ok. Redirecting to quiz: {user_info}"); return redirect(url_for('quiz', **user_info))
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
    print(f"Quiz route: User '{user_info['name']}', State '{target_state}'. Items: {len(filtered_quiz_items)}.")
    if not filtered_quiz_items: flash(f"No items found for {target_state}.", 'info')
    return render_template('quiz.html', quiz_items=filtered_quiz_items, user_info=user_info)

@app.route('/submit', methods=['POST'])
def submit():
    """Handles the submission of the quiz answers to MySQL."""
    print("--- submit route: Received POST request ---")
    cursor = get_db()
    if not cursor:
        print("Submit Error: DB connection failed (get_db returned None).")
        # FIX: Check flashed messages correctly (now that it's imported)
        if not any(msg[1] == 'error' for msg in get_flashed_messages(with_categories=True)):
             flash("Database connection failed. Cannot submit results.", "error")
        return redirect(url_for('index'))

    db_connection = getattr(g, 'db', None)
    try:
        user_name = request.form.get('user_name'); user_state = request.form.get('user_state')
        user_age_str = request.form.get('user_age'); user_sex = request.form.get('user_sex') or None
        if not user_name or not user_state:
            print("Submit Error: User info missing in form."); flash("User info missing.", 'error'); return redirect(url_for('index'))
        user_age = None
        if user_age_str:
            try: user_age = int(user_age_str); assert user_age >= 0
            except: print(f"Warn: Invalid age '{user_age_str}'."); flash("Invalid age.", "warning"); user_age = None
        print(f"Submit route: Proc User: {user_name}, State: {user_state}, Age: {user_age}, Sex: {user_sex}")
        results_to_insert = []; processed_indices = set(); item_count = 0
        for key in request.form:
             if key.startswith('annotation_'):
                 item_count += 1; identifier = key.split('_')[-1]
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
                         except: print(f"Warn: Invalid rating '{rating_str}' for item {identifier}."); offensiveness = -1
                     else: print(f"Warn: Stereotype '{identifier}' missing rating.")
                 results_to_insert.append({'user_name': user_name,'user_state': user_state,'user_age': user_age,'user_sex': user_sex,
                                           'category': category,'attribute_superset': superset,'annotation': annotation,'offensiveness_rating': offensiveness})
        print(f"Submit route: Found {item_count} items. Processed {len(processed_indices)}. Prepared {len(results_to_insert)} rows.")
        if results_to_insert:
             sql = f"INSERT INTO {RESULTS_TABLE} (user_name, user_state, user_age, user_sex, category, attribute_superset, annotation, offensiveness_rating) VALUES (%(user_name)s, %(user_state)s, %(user_age)s, %(user_sex)s, %(category)s, %(attribute_superset)s, %(annotation)s, %(offensiveness_rating)s)"
             try:
                 print(f"Submit route: Inserting {len(results_to_insert)} rows..."); cursor.executemany(sql, results_to_insert)
                 if db_connection: db_connection.commit(); print(f"Submit route: Inserted {cursor.rowcount}. Committed."); flash(f"Submitted {len(results_to_insert)} responses. Thank you!", 'success')
                 else: print("Submit Error: No DB connection to commit."); flash("Internal error.", "error"); return redirect(url_for('index'))
             except MySQLError as db_err:
                  print(f"DB INSERT ERROR: {db_err}"); print(f"Data[0]: {results_to_insert[0] if results_to_insert else 'N/A'}")
                  try:
                      if db_connection: db_connection.rollback(); print("Submit route: Rolled back.")
                  except: pass
                  flash("DB error saving responses.", 'error'); return redirect(url_for('index'))
             except Exception as e:
                  print(f"UNEXPECTED INSERT ERROR: {e}\n{traceback.format_exc()}")
                  try:
                      if db_connection: db_connection.rollback()
                  except: pass
                  flash("Unexpected error saving data.", 'error'); return redirect(url_for('index'))
        else: print("Warning: No valid results parsed."); flash("No valid responses found.", 'warning')
        print("Submit route: Redirecting to thank_you."); return redirect(url_for('thank_you'))
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

# --- Admin Routes ---
# (Keep admin_view, download_processed_data, download_raw_data exactly as before)
# Remember to add authentication!
@app.route('/admin')
def admin_view():
    print("--- admin_view: Request ---"); cursor = get_db()
    if not cursor: print("Admin Error: DB connection failed."); flash("DB connection failed.", "error"); return redirect(url_for('index'))
    results_data = []
    try:
        query = f"SELECT * FROM {RESULTS_TABLE} ORDER BY timestamp DESC"; print(f"Admin: Query: {query}"); cursor.execute(query)
        results_data = cursor.fetchall(); print(f"Admin: Fetched {len(results_data)} results.")
    except MySQLError as err:
        print(f"Admin DB Error: {err}")
        if "Unknown column 'timestamp'" in str(err):
             print("Admin Warn: timestamp column missing."); fallback_query = f"SELECT * FROM {RESULTS_TABLE}"
             try: cursor.execute(fallback_query); results_data = cursor.fetchall(); print(f"Admin: Fetched {len(results_data)} (fallback).")
             except MySQLError as inner_err: print(f"Admin DB Error (fallback): {inner_err}"); flash(f'Error fetching results: {inner_err}', 'error')
        else: flash(f'Error fetching results: {err}', 'error')
    except Exception as e: print(f"Admin Unexpected Error: {e}\n{traceback.format_exc()}"); flash('Unexpected error.', 'error')
    return render_template('admin.html', results=results_data)

@app.route('/admin/download_processed')
def download_processed_data():
    print("--- download_processed: Request ---"); aggregated_df = generate_aggregated_data()
    if aggregated_df is None: print("Download Proc Err: Processing failed."); return redirect(url_for('admin_view'))
    if aggregated_df.empty: print("Download Proc Info: Empty df."); flash("No data to process.", "warning"); return redirect(url_for('admin_view'))
    try:
        print(f"Download Proc: Generating CSV ({len(aggregated_df)} rows)..."); buffer = io.BytesIO()
        aggregated_df.to_csv(buffer, index=False, encoding='utf-8'); buffer.seek(0)
        name = 'final_aggregated_stereotypes.csv'; print(f"Download Proc: Sending '{name}'..."); return send_file(buffer, mimetype='text/csv', download_name=name, as_attachment=True)
    except Exception as e: print(f"Download Proc Err: {e}\n{traceback.format_exc()}"); flash(f"Error creating file: {e}", "error"); return redirect(url_for('admin_view'))

@app.route('/admin/download_raw')
def download_raw_data():
    print("--- download_raw: Request ---"); db_conn_raw = None
    try:
        print("[Raw DL] Connecting..."); db_conn_raw = mysql.connector.connect(host=app.config['MYSQL_HOST'], user=app.config['MYSQL_USER'], password=app.config['MYSQL_PASSWORD'], database=app.config['MYSQL_DB'], port=app.config['MYSQL_PORT'], connection_timeout=10 )
        if not db_conn_raw.is_connected(): raise MySQLError("Raw DL: Failed connection.")
        query = f"SELECT * FROM {RESULTS_TABLE} ORDER BY timestamp DESC"; print(f"[Raw DL] Query: {query}"); raw_results_df = pd.read_sql_query(query, db_conn_raw)
        print(f"[Raw DL] Fetched {len(raw_results_df)} rows.")
        if raw_results_df.empty: print("[Raw DL] Info: Empty table."); flash("Raw results empty.", "warning"); return redirect(url_for('admin_view'))
        print("[Raw DL] Generating CSV..."); buffer = io.BytesIO(); raw_results_df.to_csv(buffer, index=False, encoding='utf-8'); buffer.seek(0)
        name = 'raw_quiz_results.csv'; print(f"[Raw DL] Sending '{name}'..."); return send_file(buffer, mimetype='text/csv', download_name=name, as_attachment=True )
    except (MySQLError, pd.errors.DatabaseError) as e: print(f"ERROR [Raw DL]: DB/Pandas error: {e}"); flash(f"Error fetching raw data: {e}", "error"); return redirect(url_for('admin_view'))
    except Exception as e: print(f"UNEXPECTED ERROR [Raw DL]:\n{traceback.format_exc()}"); flash(f"Unexpected error preparing raw download: {e}", "error"); return redirect(url_for('admin_view'))
    finally:
        if db_conn_raw and db_conn_raw.is_connected(): db_conn_raw.close(); print("[Raw DL] Closed DB connection.")

# Removed the __main__ block for deployment