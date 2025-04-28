# stereotype_quiz_app/app.py

import os
import csv
import io
import mysql.connector
from mysql.connector import Error as MySQLError
# Import urlparse
from urllib.parse import urlparse # <--- ADD THIS IMPORT
# Ensure get_flashed_messages is imported
from flask import (Flask, render_template, request, redirect, url_for, g,
                   flash, Response, send_file, get_flashed_messages)
import pandas as pd
import numpy as np
import traceback

# --- Configuration ---
CSV_FILE_PATH = os.path.join('data', 'stereotypes.csv')
SCHEMA_FILE = 'schema_mysql.sql'

# --- Configuration from Environment Variables ---
SECRET_KEY = os.environ.get('SECRET_KEY', 'a_very_weak_default_secret_key_change_me_if_local')

# --- Read and Parse MySQL URL from Railway ---
MYSQL_URL = os.environ.get('MYSQL_URL') # <--- READ THE SINGLE URL VARIABLE
parsed_url = None
db_config = {} # Dictionary to hold parsed components

if MYSQL_URL:
    try:
        parsed_url = urlparse(MYSQL_URL)
        if parsed_url.scheme != 'mysql':
             raise ValueError("Invalid scheme in MYSQL_URL")

        db_config['MYSQL_HOST'] = parsed_url.hostname
        db_config['MYSQL_USER'] = parsed_url.username
        db_config['MYSQL_PASSWORD'] = parsed_url.password
        # Path includes the leading '/', remove it for database name
        db_config['MYSQL_DB'] = parsed_url.path[1:] if parsed_url.path else None
        db_config['MYSQL_PORT'] = parsed_url.port

        # Validate essential components
        if not all([db_config['MYSQL_HOST'], db_config['MYSQL_USER'], db_config['MYSQL_DB'], db_config['MYSQL_PORT'] is not None]):
             print(f"CRITICAL ERROR: MYSQL_URL parsed incompletely. Missing components in: {MYSQL_URL}")
             db_config = {} # Reset config if incomplete
        else:
             print("--- Application Configuration (Reading MYSQL_URL) ---")
             print(f"SECRET_KEY: {'Set' if SECRET_KEY != 'a_very_weak_default_secret_key_change_me_if_local' else 'Default (UNSAFE!)'}")
             print(f"MYSQL_HOST: {db_config.get('MYSQL_HOST')}")
             print(f"MYSQL_USER: {db_config.get('MYSQL_USER')}")
             print(f"MYSQL_PASSWORD: {'Set' if db_config.get('MYSQL_PASSWORD') else 'Not Set'}")
             print(f"MYSQL_DB: {db_config.get('MYSQL_DB')}")
             print(f"MYSQL_PORT: {db_config.get('MYSQL_PORT')}")
             print("----------------------------------------------------")

    except Exception as e:
        print(f"CRITICAL ERROR: Failed to parse MYSQL_URL: {MYSQL_URL}. Error: {e}")
        db_config = {} # Ensure config is empty on error
else:
    print("CRITICAL ERROR: MYSQL_URL environment variable not found.")


RESULTS_TABLE = 'results'

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY

# Populate app.config ONLY if essential components were found from URL parsing
if db_config:
    app.config.update(db_config) # Update app.config with parsed values
    app.config['DB_CONFIGURED'] = True
else:
    print("--- Application Configuration FAILED: Could not setup DB from MYSQL_URL ---")
    app.config['DB_CONFIGURED'] = False


# --- Database Functions ---
def get_db():
    """Opens a new MySQL database connection and cursor if none exist for the current request context."""
    if 'db' not in g:
        if not app.config.get('DB_CONFIGURED', False): # Check our flag
             print("ERROR get_db: Database configuration failed during startup (MYSQL_URL issue).")
             # Avoid flashing here, might happen before first request context
             # flash('Database configuration error. Please contact admin.', 'error')
             g.db = None; g.cursor = None
             return None

        # Ensure all keys needed for connection exist in app.config
        required_keys = ['MYSQL_HOST', 'MYSQL_USER', 'MYSQL_PASSWORD', 'MYSQL_DB', 'MYSQL_PORT']
        if not all(k in app.config for k in required_keys):
             print(f"ERROR get_db: Database configuration missing required keys in app.config. Keys found: {list(app.config.keys())}")
             # flash('Database configuration error. Please contact admin.', 'error') # Avoid flashing here
             g.db = None; g.cursor = None
             return None

        try:
            # Use the keys populated from the parsed URL
            g.db = mysql.connector.connect(
                host=app.config['MYSQL_HOST'],
                user=app.config['MYSQL_USER'],
                password=app.config['MYSQL_PASSWORD'], # Can be None if not in URL
                database=app.config['MYSQL_DB'],
                port=app.config['MYSQL_PORT'],
                connection_timeout=10 # Keep timeout
            )
            g.cursor = g.db.cursor(dictionary=True)
            print(f"DEBUG get_db: Successfully connected to {app.config['MYSQL_HOST']}:{app.config['MYSQL_PORT']}")
        except MySQLError as err:
            # More detailed error logging
            print(f"ERROR connecting to MySQL: {err}")
            print(f"DEBUG: Connection attempt failed with Host={app.config.get('MYSQL_HOST')}, User={app.config.get('MYSQL_USER')}, DB={app.config.get('MYSQL_DB')}, Port={app.config.get('MYSQL_PORT')}")
            flash('Database connection error. Please try again later or contact admin.', 'error')
            g.db = None; g.cursor = None
        except Exception as e: # Catch other potential errors
            print(f"UNEXPECTED ERROR in get_db during connect: {e}")
            print(traceback.format_exc())
            flash('An unexpected error occurred connecting to the database.', 'error')
            g.db = None; g.cursor = None

    return getattr(g, 'cursor', None)

# --- Teardown remains the same ---
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

# --- init_db now uses app.config populated by MYSQL_URL ---
def init_db():
    """Connects to MySQL, creates the database IF NOT EXISTS, and creates the table IF NOT EXISTS using schema_mysql.sql."""
    if not app.config.get('DB_CONFIGURED', False):
         print("CRITICAL init_db ERROR: Database configuration missing (MYSQL_URL issue). Skipping initialization.")
         return

    # The rest of init_db should work as it reads from app.config,
    # which is now populated correctly from the parsed MYSQL_URL.
    # No changes needed inside the try/except block of init_db itself,
    # just ensure it reads from app.config correctly.

    # Double-check required keys before proceeding inside init_db for safety
    required_keys = ['MYSQL_HOST', 'MYSQL_USER', 'MYSQL_PASSWORD', 'MYSQL_DB', 'MYSQL_PORT']
    if not all(k in app.config for k in required_keys):
        print(f"CRITICAL init_db ERROR: Missing required keys in app.config. Cannot proceed.")
        return

    temp_conn = None; temp_cursor = None
    db_host = app.config['MYSQL_HOST']; db_user = app.config['MYSQL_USER']
    db_password = app.config['MYSQL_PASSWORD']; db_port = app.config['MYSQL_PORT']
    db_name = app.config['MYSQL_DB']
    print(f"--- init_db: Attempting DB Initialization (Host: {db_host}, User: {db_user}, DB: {db_name}, Port: {db_port}) ---")
    try:
        # Connect to the server *without* specifying a database first to create it
        print(f"init_db: Connecting to MySQL server ({db_host}:{db_port}) to check/create database...")
        temp_conn = mysql.connector.connect(
            host=db_host, user=db_user, password=db_password, port=db_port, connection_timeout=15
        )
        temp_cursor = temp_conn.cursor()
        print(f"init_db: Connected to MySQL server. Checking/creating database '{db_name}'...")
        # Create DB if not exists
        try:
            # Use backticks for safety with potentially reserved keywords
            temp_cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            temp_conn.commit()
            print(f"init_db: Database '{db_name}' checked/created.")
        except MySQLError as err:
            # This might fail due to permissions, which can be okay if DB already exists
            print(f"Warning init_db: Could not CREATE DATABASE (might be permissions or DB exists): {err}.")

        # Switch to the specific database
        try:
            temp_cursor.execute(f"USE `{db_name}`")
            print(f"init_db: Switched to database '{db_name}'.")
        except MySQLError as err:
            print(f"CRITICAL init_db ERROR: Failed to switch to database '{db_name}': {err}. Check if DB exists and user has privileges.")
            # Can't proceed if we can't USE the database
            return

        # Check for table and execute schema if needed
        print(f"init_db: Checking for table '{RESULTS_TABLE}'...");
        temp_cursor.execute(f"SHOW TABLES LIKE '{RESULTS_TABLE}'")
        table_exists = temp_cursor.fetchone()
        if not table_exists:
             print(f"init_db: Table '{RESULTS_TABLE}' not found. Creating table(s) from schema...")
             # Locate schema file relative to app root
             schema_path = os.path.join(app.root_path, SCHEMA_FILE)
             # Fallback check if SCHEMA_FILE doesn't contain directory info
             if not os.path.exists(schema_path):
                 schema_path_alt = os.path.join(app.root_path, 'schema_mysql.sql')
                 if os.path.exists(schema_path_alt):
                     schema_path = schema_path_alt
                 else:
                     # Critical error if schema file cannot be found
                     raise FileNotFoundError(f"init_db: Schema file not found at expected locations: {os.path.join(app.root_path, SCHEMA_FILE)} or {schema_path_alt}")

             try:
                 with open(schema_path, mode='r', encoding='utf-8') as f:
                     sql_script = f.read()
                 print(f"init_db: Executing SQL script from {schema_path}...")
                 # Execute potentially multiple statements in the script
                 statement_results = temp_cursor.execute(sql_script, multi=True)
                 # Iterate through results to ensure execution and consume potential result sets
                 for i, result in enumerate(statement_results):
                     # print(f"  Statement {i+1}: Rows affected: {result.rowcount}")
                     if result.with_rows:
                         # print(f"  Statement {i+1} produced rows, fetching...")
                         result.fetchall() # Consume results to avoid errors
                 temp_conn.commit()
                 print(f"init_db: Database table(s) created/updated from schema.")
             except FileNotFoundError as e: # Catch specific error from above
                 print(f"CRITICAL init_db ERROR: Schema file missing: {e}.")
             except MySQLError as e:
                 print(f"CRITICAL init_db ERROR executing schema SQL: {e}")
                 temp_conn.rollback() # Rollback changes on error
             except Exception as e:
                 print(f"CRITICAL init_db UNEXPECTED ERROR executing schema: {e}\n{traceback.format_exc()}")
                 temp_conn.rollback()
        else:
             print(f"init_db: Database table '{RESULTS_TABLE}' already exists.")

    except MySQLError as e:
         print(f"CRITICAL init_db ERROR during connection/setup: {e}")
    except Exception as e:
         print(f"CRITICAL init_db UNEXPECTED error during initialization: {e}\n{traceback.format_exc()}")
    finally:
        # Ensure cursor and connection are closed
        if temp_cursor:
            try: temp_cursor.close()
            except Exception as e_close: print(f"Warning: Error closing init_db cursor: {e_close}")
        if temp_conn and temp_conn.is_connected():
            try: temp_conn.close()
            except Exception as e_close: print(f"Warning: Error closing init_db connection: {e_close}")
        print("--- init_db: Finished DB Initialization Check ---")


# --- Initialize DB on Application Start ---
# Check the DB_CONFIGURED flag set during startup
if app.config.get('DB_CONFIGURED'):
    print(">>> Application starting: Performing database initialization check...")
    with app.app_context(): init_db()
    print(">>> Application starting: Database initialization check complete.")
else:
    print(">>> Application starting: Skipping DB initialization due to configuration error (MYSQL_URL).")


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
# (generate_aggregated_data needs to use app.config populated by MYSQL_URL)
def generate_aggregated_data():
    # print("--- [Processing] Starting data aggregation ---") # Less verbose
    db_conn_proc = None; aggregated_df = None
    try:
        # Check the DB_CONFIGURED flag first
        if not app.config.get('DB_CONFIGURED'):
             raise ValueError("Processing Error: Database configuration not available (MYSQL_URL issue).")
        # Check required keys exist in app.config
        required_keys = ['MYSQL_HOST', 'MYSQL_USER', 'MYSQL_PASSWORD', 'MYSQL_DB', 'MYSQL_PORT']
        if not all(k in app.config for k in required_keys):
            raise ValueError(f"Processing Error: Database configuration missing required keys in app.config.")

        # print("[Processing] Connecting to DB...") # Less verbose
        db_conn_proc = mysql.connector.connect(
            host=app.config['MYSQL_HOST'], user=app.config['MYSQL_USER'],
            password=app.config['MYSQL_PASSWORD'], database=app.config['MYSQL_DB'],
            port=app.config['MYSQL_PORT'], connection_timeout=10)

        if not db_conn_proc.is_connected(): raise MySQLError("Processing: Failed connection.")

        # Rest of the function remains the same as it reads data using the connection
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

# --- calculation function remains the same ---
def calculate_mean_offensiveness(series):
    valid_ratings = series[series >= 0]; return valid_ratings.mean() if not valid_ratings.empty else np.nan

# --- Flask Routes ---
# (Routes index, quiz, submit, thank_you, admin_view use get_db(), so they benefit from the fix)
# (Routes download_processed_data, download_raw_data use generate_aggregated_data() or connect directly,
#  so they also benefit from the fix via app.config)

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
    # Check if DB configured before rendering
    if not app.config.get('DB_CONFIGURED', False):
         flash("Application database is not configured. Please contact administrator.", "error")
         # Optionally render a simpler page or just return the message
         # return "Database configuration error", 500
    return render_template('index.html', states=INDIAN_STATES, form_data={})

@app.route('/quiz')
def quiz():
    user_info = {'name': request.args.get('name'),'state': request.args.get('state'),'age': request.args.get('age'),'sex': request.args.get('sex')}
    if not user_info['name'] or not user_info['state']:
         flash('User info missing. Please start again.', 'error'); return redirect(url_for('index'))
    if not ALL_STEREOTYPE_DATA or (INDIAN_STATES and INDIAN_STATES[0].startswith("Error:")):
        flash('Error: Stereotype data not loaded.', 'error'); return redirect(url_for('index'))
    # Check DB config before proceeding if quiz depends on DB state (it doesn't seem to directly here)
    # if not app.config.get('DB_CONFIGURED', False):
    #      flash("Database not available.", "error"); return redirect(url_for('index'))
    target_state = user_info['state']
    filtered_quiz_items = [item for item in ALL_STEREOTYPE_DATA if item['state'] == target_state]
    if not filtered_quiz_items: flash(f"No items found for {target_state}.", 'info')
    return render_template('quiz.html', quiz_items=filtered_quiz_items, user_info=user_info)

@app.route('/submit', methods=['POST'])
def submit():
    cursor = get_db() # get_db() now handles connection logic based on MYSQL_URL
    if not cursor:
        # get_db() already logs the error and might flash a message
        print("Submit Error: get_db() failed to provide a cursor.")
        # Redirecting to index where a flash message might be shown by get_db's failure handling
        return redirect(url_for('index'))

    db_connection = getattr(g, 'db', None) # Get the connection object stored in g by get_db()
    if not db_connection or not db_connection.is_connected():
        print("Submit Error: Database connection object not found or not connected in 'g'.")
        flash("Internal server error: Database connection lost.", "error")
        return redirect(url_for('index'))

    try:
        user_name = request.form.get('user_name'); user_state = request.form.get('user_state')
        user_age_str = request.form.get('user_age'); user_sex = request.form.get('user_sex') or None
        if not user_name or not user_state:
            flash("User info missing from submission.", 'error'); return redirect(url_for('index'))
        user_age = None
        if user_age_str and user_age_str.isdigit():
            try: user_age = int(user_age_str); assert user_age >= 0
            except (ValueError, AssertionError): flash("Invalid age provided.", "warning"); user_age = None
        elif user_age_str: # If it's not empty and not digits
             flash("Invalid age format.", "warning"); user_age = None

        results_to_insert = []; processed_indices = set()
        # Loop through form data to find annotations
        for key in request.form:
             if key.startswith('annotation_'):
                 # Extract the identifier (expecting digits)
                 parts = key.split('_')
                 if len(parts) > 1 and parts[-1].isdigit():
                     identifier = parts[-1]
                     # Avoid processing the same item twice if form keys are weird
                     if identifier in processed_indices: continue
                     processed_indices.add(identifier)

                     # Retrieve related fields using the identifier
                     superset = request.form.get(f'superset_{identifier}')
                     category = request.form.get(f'category_{identifier}')
                     annotation = request.form.get(key) # The annotation value itself

                     # Basic validation
                     if not all([superset, category, annotation]):
                         print(f"Warning: Missing data for item identifier {identifier}. Skipping.")
                         continue

                     offensiveness = -1 # Default for non-stereotype or missing rating
                     if annotation == 'Stereotype':
                         rating_str = request.form.get(f'offensiveness_{identifier}')
                         if rating_str is not None and rating_str.isdigit():
                             try:
                                 offensiveness_val = int(rating_str)
                                 if 0 <= offensiveness_val <= 5:
                                     offensiveness = offensiveness_val
                                 else:
                                     print(f"Warning: Invalid offensiveness rating '{rating_str}' for item {identifier}. Using -1.")
                             except ValueError:
                                 print(f"Warning: Could not parse offensiveness rating '{rating_str}' for item {identifier}. Using -1.")
                         elif rating_str: # If present but not digits
                             print(f"Warning: Non-digit offensiveness rating '{rating_str}' for item {identifier}. Using -1.")

                     # Append data for insertion
                     results_to_insert.append({
                         'user_name': user_name, 'user_state': user_state,
                         'user_age': user_age, 'user_sex': user_sex,
                         'category': category, 'attribute_superset': superset,
                         'annotation': annotation, 'offensiveness_rating': offensiveness
                     })
                 else:
                      print(f"Warning: Malformed form key encountered: {key}")


        if results_to_insert:
             sql = f"""
                 INSERT INTO {RESULTS_TABLE}
                 (user_name, user_state, user_age, user_sex, category, attribute_superset, annotation, offensiveness_rating)
                 VALUES
                 (%(user_name)s, %(user_state)s, %(user_age)s, %(user_sex)s, %(category)s, %(attribute_superset)s, %(annotation)s, %(offensiveness_rating)s)
             """
             try:
                 cursor.executemany(sql, results_to_insert)
                 db_connection.commit() # Commit the transaction
                 flash(f"Successfully submitted {len(results_to_insert)} responses. Thank you!", 'success')
             except MySQLError as db_err:
                  print(f"DB INSERT ERROR: {db_err}")
                  print(f"Data attempted: {results_to_insert[:2]}...") # Log first few records
                  try:
                      db_connection.rollback() # Rollback on error
                  except Exception as rb_err:
                      print(f"Error during rollback: {rb_err}")
                  flash("Database error saving responses. Please try again.", 'error');
                  # Decide if redirecting to index or quiz is better
                  return redirect(url_for('index')) # Or potentially back to quiz?
             except Exception as e:
                  print(f"UNEXPECTED INSERT ERROR: {e}\n{traceback.format_exc()}")
                  try:
                      db_connection.rollback()
                  except Exception as rb_err:
                      print(f"Error during rollback: {rb_err}")
                  flash("An unexpected error occurred while saving data.", 'error')
                  return redirect(url_for('index'))
        else:
             flash("No valid responses were processed from your submission.", 'warning')
             # If nothing was submitted, maybe redirect back to quiz? Or thank_you anyway?
             # return redirect(url_for('quiz', name=user_name, state=user_state, age=user_age, sex=user_sex))

        return redirect(url_for('thank_you'))

    except Exception as e:
        # Catch-all for unexpected errors in the route logic itself
        print(f"SUBMIT ROUTE UNEXPECTED ERROR: {e}\n{traceback.format_exc()}");
        flash("An unexpected error occurred during submission processing.", 'error')
        # Attempt rollback if connection exists
        db_conn = getattr(g, 'db', None)
        if db_conn and db_conn.is_connected():
            try: db_conn.rollback()
            except Exception as rb_err: print(f"Rollback attempt failed in outer catch: {rb_err}")
        return redirect(url_for('index'))


@app.route('/thank_you')
def thank_you():
    return render_template('thank_you.html')

@app.route('/admin')
def admin_view():
    cursor = get_db() # Uses the corrected get_db
    if not cursor:
        flash("Database connection failed. Cannot display admin view.", "error")
        return redirect(url_for('index')) # Or render an admin error page

    results_data = []
    try:
        # Ensure RESULTS_TABLE is defined correctly
        query = f"SELECT * FROM {RESULTS_TABLE} ORDER BY timestamp DESC"
        cursor.execute(query)
        results_data = cursor.fetchall() # Fetch all results as dictionaries
    except MySQLError as err:
        print(f"Admin View DB Error: {err}")
        flash(f'Error fetching results: {err}', 'error')
    except NameError: # If RESULTS_TABLE wasn't defined somehow
         print("Admin View Error: RESULTS_TABLE variable not defined.")
         flash('Configuration error: Results table name missing.', 'error')
    except Exception as e:
        print(f"Admin View Unexpected Error: {e}\n{traceback.format_exc()}")
        flash('An unexpected error occurred fetching admin data.', 'error')

    return render_template('admin.html', results=results_data)


@app.route('/admin/download_processed')
def download_processed_data():
    # generate_aggregated_data now uses the correct connection info via app.config
    aggregated_df = generate_aggregated_data()

    # Handle cases where aggregation fails or returns empty
    if aggregated_df is None:
        # Flash message might already be set by generate_aggregated_data
        if not get_flashed_messages(category_filter=["error"]):
            flash("Failed to generate processed data due to an error.", "error")
        return redirect(url_for('admin_view'))

    if aggregated_df.empty:
        flash("No data available to process and download.", "warning")
        return redirect(url_for('admin_view'))

    try:
        # Use BytesIO for in-memory file handling
        buffer = io.BytesIO()
        aggregated_df.to_csv(buffer, index=False, encoding='utf-8')
        buffer.seek(0) # Rewind buffer to the beginning
        download_name = 'final_aggregated_stereotypes.csv'
        return send_file(
            buffer,
            mimetype='text/csv',
            download_name=download_name,
            as_attachment=True
        )
    except Exception as e:
        print(f"Download Processed Data Error: {e}\n{traceback.format_exc()}")
        flash(f"Error creating processed data file: {e}", "error")
        return redirect(url_for('admin_view'))


@app.route('/admin/download_raw')
def download_raw_data():
    db_conn_raw = None
    try:
        # Check config status first
        if not app.config.get('DB_CONFIGURED'):
             raise ValueError("Raw Download Error: Database configuration not available (MYSQL_URL issue).")
        # Check required keys
        required_keys = ['MYSQL_HOST', 'MYSQL_USER', 'MYSQL_PASSWORD', 'MYSQL_DB', 'MYSQL_PORT']
        if not all(k in app.config for k in required_keys):
            raise ValueError("Raw Download Error: Database configuration missing required keys.")

        # Connect using app.config values derived from MYSQL_URL
        db_conn_raw = mysql.connector.connect(
            host=app.config['MYSQL_HOST'], user=app.config['MYSQL_USER'],
            password=app.config['MYSQL_PASSWORD'], database=app.config['MYSQL_DB'],
            port=app.config['MYSQL_PORT'], connection_timeout=10
        )

        if not db_conn_raw.is_connected():
            raise MySQLError("Raw Download: Failed to establish database connection.")

        query = f"SELECT * FROM {RESULTS_TABLE} ORDER BY timestamp DESC"
        # Use pandas to read directly into DataFrame
        raw_results_df = pd.read_sql_query(query, db_conn_raw)

        if raw_results_df.empty:
            flash("No raw results found in the database.", "warning")
            return redirect(url_for('admin_view'))

        # Create CSV in memory
        buffer = io.BytesIO()
        raw_results_df.to_csv(buffer, index=False, encoding='utf-8')
        buffer.seek(0)
        download_name = 'raw_quiz_results.csv'
        return send_file(
            buffer,
            mimetype='text/csv',
            download_name=download_name,
            as_attachment=True
        )

    # Catch specific expected errors
    except (MySQLError, pd.errors.DatabaseError) as e:
        print(f"ERROR [Raw Download DB]: {e}")
        flash(f"Database error fetching raw data: {e}", "error")
        return redirect(url_for('admin_view'))
    except ValueError as e: # Catch config errors
         print(f"ERROR [Raw Download Config]: {e}")
         flash(f"Configuration error for raw download: {e}", "error")
         return redirect(url_for('admin_view'))
    # Catch any other unexpected errors
    except Exception as e:
        print(f"UNEXPECTED ERROR [Raw Download]:\n{traceback.format_exc()}")
        flash(f"An unexpected error occurred preparing the raw data download: {e}", "error")
        return redirect(url_for('admin_view'))
    finally:
        # Ensure the connection is closed
        if db_conn_raw and db_conn_raw.is_connected():
            try: db_conn_raw.close()
            except Exception as e_close: print(f"Warning: Error closing raw download connection: {e_close}")


# Removed the __main__ block for deployment
# if __name__ == '__main__':
#     app.run(debug=True) # Debug should be False in production