# -*- coding: utf-8 -*-
import os
import csv
import io
import mysql.connector
from mysql.connector import Error as MySQLError
from flask import Flask, render_template, request, redirect, url_for, g, flash, Response, send_file, current_app
import pandas as pd
import numpy as np
from dotenv import load_dotenv # To load .env file for local development
import logging # Import logging
import sys # For potentially exiting on critical errors
import traceback # Import traceback for detailed error logging
# No longer need quote_plus as we won't build connection strings for pandas here

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
CSV_FILE_PATH = os.path.join('data', 'stereotypes.csv')
SCHEMA_FILE = 'schema.sql' # User confirmed filename

# Flask Secret Key
SECRET_KEY = os.environ.get('SECRET_KEY')
if not SECRET_KEY:
    print("CRITICAL ERROR: SECRET_KEY is not set in environment variables.")
    SECRET_KEY = 'insecure-default-key-set-in-environment'
    print("WARNING: Using insecure default SECRET_KEY!")

# --- MySQL Configuration ---
MYSQL_HOST = os.environ.get('MYSQL_HOST')
MYSQL_USER = os.environ.get('MYSQL_USER')
MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD')
MYSQL_DB = os.environ.get('MYSQL_DB')
MYSQL_PORT = int(os.environ.get('MYSQL_PORT', 3306))
RESULTS_TABLE = 'results'

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['MYSQL_HOST'] = MYSQL_HOST
app.config['MYSQL_USER'] = MYSQL_USER
app.config['MYSQL_PASSWORD'] = MYSQL_PASSWORD
app.config['MYSQL_DB'] = MYSQL_DB
app.config['MYSQL_PORT'] = MYSQL_PORT

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
app.logger.setLevel(logging.INFO)

app.logger.info("Flask application starting...")
if not all([MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB]):
    app.logger.critical("CRITICAL STARTUP ERROR: Database configuration missing in environment variables.")
else:
    app.logger.info(f"Database configuration loaded: Host={MYSQL_HOST}, User={MYSQL_USER}, DB={MYSQL_DB}, Port={MYSQL_PORT}")

# --- Database Functions ---
# get_db() and close_db() remain the same as they work correctly
def get_db():
    if 'db' not in g:
        db_host = current_app.config['MYSQL_HOST']; db_user = current_app.config['MYSQL_USER']
        db_password = current_app.config['MYSQL_PASSWORD']; db_name = current_app.config['MYSQL_DB']
        db_port = current_app.config['MYSQL_PORT']
        if not all([db_host, db_user, db_password, db_name]):
            app.logger.error("get_db failed: DB configuration incomplete in app.config.")
            return None
        try:
            app.logger.debug(f"Attempting DB connection to {db_user}@{db_host}:{db_port}/{db_name}")
            g.db = mysql.connector.connect(host=db_host, user=db_user, password=db_password, database=db_name, port=db_port, connection_timeout=10)
            g.cursor = g.db.cursor(dictionary=True) # Use dictionary cursor
            app.logger.debug("MySQL connection established.")
        except MySQLError as err:
            app.logger.error(f"Error connecting to MySQL: {err}", exc_info=True)
            flash('Database connection error. Please check server logs.', 'error')
            g.db = None; g.cursor = None
    return getattr(g, 'cursor', None)

@app.teardown_appcontext
def close_db(error):
    cursor = g.pop('cursor', None)
    if cursor:
        try: cursor.close()
        except Exception as e: app.logger.warning(f"Error closing cursor: {e}")
    db = g.pop('db', None)
    if db and db.is_connected():
        try: db.close(); app.logger.debug("MySQL connection closed.")
        except Exception as e: app.logger.warning(f"Error closing DB connection: {e}")
    if error: app.logger.error(f"App context teardown error: {error}", exc_info=True)

# init_db_table() remains the same
def init_db_table():
    db_host = current_app.config['MYSQL_HOST']; db_user = current_app.config['MYSQL_USER']
    db_password = current_app.config['MYSQL_PASSWORD']; db_name = current_app.config['MYSQL_DB']
    db_port = current_app.config['MYSQL_PORT']
    if not all([db_host, db_user, db_password, db_name]):
        app.logger.error("Cannot initialize DB table: Environment configuration missing.")
        return False
    db_conn_init = None; cursor_init = None
    try:
        app.logger.info(f"Connecting to DB '{db_name}' for table init check...")
        db_conn_init = mysql.connector.connect(host=db_host, user=db_user, password=db_password, database=db_name, port=db_port)
        cursor_init = db_conn_init.cursor()
        app.logger.info(f"Checking for table '{RESULTS_TABLE}'...")
        cursor_init.execute(f"SHOW TABLES LIKE '{RESULTS_TABLE}'")
        if not cursor_init.fetchone():
            app.logger.warning(f"Table '{RESULTS_TABLE}' not found. Creating using '{SCHEMA_FILE}'...")
            schema_path = os.path.join(current_app.root_path, SCHEMA_FILE)
            if not os.path.exists(schema_path):
                 app.logger.error(f"Schema file '{SCHEMA_FILE}' not found at: {schema_path}"); return False
            try:
                with open(schema_path, mode='r', encoding='utf-8') as f: sql_script = f.read()
                app.logger.info(f"Executing SQL script from {schema_path}...")
                for result in cursor_init.execute(sql_script, multi=True):
                     app.logger.debug(f"Schema exec result: Stmt='{result.statement}', Rows={result.rowcount}")
                     if result.with_rows: result.fetchall()
                db_conn_init.commit()
                app.logger.info(f"Table '{RESULTS_TABLE}' created successfully.")
                return True
            except MySQLError as err: app.logger.error(f"Error executing schema '{schema_path}': {err}", exc_info=True); db_conn_init.rollback(); return False
            except Exception as e: app.logger.error(f"Unexpected error initializing schema: {e}", exc_info=True); db_conn_init.rollback(); return False
        else:
            app.logger.info(f"Table '{RESULTS_TABLE}' already exists.")
            return True
    except MySQLError as err: app.logger.error(f"Error during DB connection/check for init: {err}", exc_info=True); return False
    except Exception as e: app.logger.error(f"Unexpected error during DB init: {e}", exc_info=True); return False
    finally:
        if cursor_init: cursor_init.close()
        if db_conn_init and db_conn_init.is_connected(): db_conn_init.close()
        app.logger.debug("DB init connection closed.")

# load_stereotype_data() remains the same
def load_stereotype_data(relative_filepath=CSV_FILE_PATH):
    stereotype_data = []
    full_filepath = os.path.join(current_app.root_path, relative_filepath)
    app.logger.info(f"Attempting to load stereotype data from: {full_filepath}")
    try:
        if not os.path.exists(full_filepath): raise FileNotFoundError(f"{full_filepath}")
        with open(full_filepath, mode='r', encoding='utf-8-sig') as infile:
            reader = csv.DictReader(infile)
            required_cols = ['State', 'Category', 'Superset', 'Subsets']
            if not reader.fieldnames or not all(field in reader.fieldnames for field in required_cols):
                 missing = [c for c in required_cols if c not in (reader.fieldnames or [])]
                 raise ValueError(f"CSV missing required columns: {missing}. Found: {reader.fieldnames}")
            for i, row in enumerate(reader):
                try:
                    state = row.get('State','').strip(); category = row.get('Category','Uncategorized').strip()
                    superset = row.get('Superset','').strip(); subsets_str = row.get('Subsets','')
                    if not state or not superset: continue
                    subsets = sorted([s.strip() for s in subsets_str.split(',') if s.strip()])
                    stereotype_data.append({'state': state, 'category': category, 'superset': superset, 'subsets': subsets})
                except Exception as row_err: app.logger.warning(f"Error processing CSV row {i+1}: {row_err}")
            app.logger.info(f"Loaded {len(stereotype_data)} stereotype entries.")
            return stereotype_data
    except FileNotFoundError as e: app.logger.critical(f"FATAL: Stereotype CSV not found: {e}"); return []
    except ValueError as ve: app.logger.critical(f"FATAL: Error processing stereotype CSV: {ve}"); return []
    except Exception as e: app.logger.critical(f"FATAL: Error loading stereotype data: {e}", exc_info=True); return []

with app.app_context():
    if not init_db_table(): # Ensure table exists before loading data or checking states
        app.logger.critical("CRITICAL: Failed to initialize or verify database table. Shutting down possible.")
        # Depending on deployment, sys.exit(1) might be appropriate here
    ALL_STEREOTYPE_DATA = load_stereotype_data()
    if ALL_STEREOTYPE_DATA:
        INDIAN_STATES = sorted(list(set(item['state'] for item in ALL_STEREOTYPE_DATA)))
        app.logger.info(f"States loaded ({len(INDIAN_STATES)} unique).")
    else:
        app.logger.error("CRITICAL: Stereotype data loading failed!")
        INDIAN_STATES = ["Error: State Data Load Failed"] # Use specific error value

# --- Data Processing Logic ---
def calculate_mean_offensiveness(series):
    valid_ratings = series[series >= 0]
    return valid_ratings.mean() if not valid_ratings.empty else np.nan

# --- REFACTORED generate_aggregated_data ---
def generate_aggregated_data():
    app.logger.info("--- [Processing] Starting data aggregation ---")
    db_conn_proc = None
    cursor_proc = None
    raw_results_list = [] # To store fetched data

    try:
        # --- Fetch raw data using direct connection ---
        db_host = current_app.config['MYSQL_HOST']
        db_user = current_app.config['MYSQL_USER']
        db_password = current_app.config['MYSQL_PASSWORD']
        db_name = current_app.config['MYSQL_DB']
        db_port = current_app.config['MYSQL_PORT']

        if not all([db_host, db_user, db_password, db_name]):
            app.logger.error("[Processing] Aborted: Database configuration missing.")
            flash("Processing Error: Server configuration issue.", "error")
            return None

        app.logger.info(f"[Processing] Attempting direct DB connection: User={db_user}, Host={db_host}, Port={db_port}, DB={db_name}")
        db_conn_proc = mysql.connector.connect(
            host=db_host, user=db_user, password=db_password, database=db_name, port=db_port, connection_timeout=10
        )
        cursor_proc = db_conn_proc.cursor(dictionary=True) # Fetch as dictionaries

        app.logger.info("[Processing] Fetching raw results via direct cursor...")
        cursor_proc.execute(f"SELECT * FROM {RESULTS_TABLE}")
        raw_results_list = cursor_proc.fetchall()
        app.logger.info(f"[Processing] Fetched {len(raw_results_list)} raw results.")

        # --- Convert fetched list to Pandas DataFrame ---
        if not raw_results_list:
            app.logger.info("[Processing] No raw results found in DB.")
            return pd.DataFrame() # Return empty DataFrame if no results

        results_df = pd.DataFrame(raw_results_list)
        app.logger.info("[Processing] Converted fetched results to DataFrame.")

        # --- Continue with existing processing logic using the DataFrame ---
        if 'user_state' not in results_df.columns: raise ValueError("Missing 'user_state' column in DB results.")
        results_df['Stereotype_State'] = results_df['user_state']

        stereotypes_path = os.path.join(current_app.root_path, CSV_FILE_PATH)
        if not os.path.exists(stereotypes_path): raise FileNotFoundError(f"{stereotypes_path}")
        app.logger.info(f"[Processing] Loading definitions from: {stereotypes_path}")
        stereotypes_df = pd.read_csv(stereotypes_path, encoding='utf-8-sig')
        req_stereo_cols = ['State', 'Category', 'Superset', 'Subsets']
        if not all(col in stereotypes_df.columns for col in req_stereo_cols):
             raise ValueError(f"Stereotypes CSV missing: {[c for c in req_stereo_cols if c not in stereotypes_df.columns]}")

        stereotypes_df['Subsets_List'] = stereotypes_df['Subsets'].fillna('').astype(str).apply(lambda x: [s.strip() for s in x.split(',') if s.strip()])
        subset_lookup = stereotypes_df.set_index(['State', 'Category', 'Superset'])['Subsets_List'].to_dict()

        app.logger.info("[Processing] Expanding annotations...")
        expanded_rows = []
        required_result_cols = ['Stereotype_State', 'category', 'attribute_superset', 'annotation', 'offensiveness_rating']
        if not all(col in results_df.columns for col in required_result_cols):
             # Check original column names from fetchall dictionary keys if needed
             original_cols = raw_results_list[0].keys() if raw_results_list else []
             app.logger.error(f"[Processing] DataFrame missing columns. Required: {required_result_cols}. Available: {list(results_df.columns)}. Original DB Cols: {list(original_cols)}")
             raise ValueError(f"Raw results missing expected columns after DataFrame conversion.")

        for index, result_row in results_df.iterrows():
            state = result_row.get('Stereotype_State'); category = result_row.get('category'); superset = result_row.get('attribute_superset')
            annotation = result_row.get('annotation'); rating = result_row.get('offensiveness_rating')

            # Check for None values which might cause issues later
            if state is None or category is None or superset is None or annotation is None:
                 app.logger.warning(f"[Processing] Skipping row {index} due to None value in essential columns (State/Cat/Super/Anno).")
                 continue
            # Ensure string types where expected
            if not all(isinstance(x, str) for x in [state, category, superset, annotation]):
                 app.logger.warning(f"[Processing] Skipping row {index} due to unexpected data type. State:'{state}', Cat:'{category}', Super:'{superset}', Anno:'{annotation}'")
                 continue
            # Handle potential None rating before adding (though DB default is -1)
            rating = rating if rating is not None else -1

            expanded_rows.append({'Stereotype_State': state, 'Category': category, 'Attribute': superset, 'annotation': annotation, 'offensiveness_rating': rating})
            for subset in subset_lookup.get((state, category, superset), []):
                expanded_rows.append({'Stereotype_State': state, 'Category': category, 'Attribute': subset, 'annotation': annotation, 'offensiveness_rating': rating})

        if not expanded_rows:
            app.logger.info("[Processing] No valid rows after expansion.")
            return pd.DataFrame()
        expanded_annotations_df = pd.DataFrame(expanded_rows)
        app.logger.info(f"[Processing] Created {len(expanded_annotations_df)} expanded rows.")

        app.logger.info("[Processing] Aggregating results...")
        grouped = expanded_annotations_df.groupby(['Stereotype_State', 'Category', 'Attribute'])
        aggregated_data = grouped.agg(
            Stereotype_Votes=('annotation', lambda x: (x == 'Stereotype').sum()),
            Not_Stereotype_Votes=('annotation', lambda x: (x == 'Not a Stereotype').sum()),
            Not_Sure_Votes=('annotation', lambda x: (x == 'Not sure').sum()),
            Average_Offensiveness=('offensiveness_rating', calculate_mean_offensiveness)
        ).reset_index()
        aggregated_data['Average_Offensiveness'] = aggregated_data['Average_Offensiveness'].round(2)
        app.logger.info(f"[Processing] Aggregation complete ({len(aggregated_data)} rows).")
        return aggregated_data

    except FileNotFoundError as e: app.logger.error(f"[Processing] File not found: {e}"); flash(f"Error: Input file missing. {e}", "error"); return None
    except MySQLError as e: app.logger.error(f"[Processing] DB Error during direct fetch/processing: {e}", exc_info=True); flash(f"Error: Database processing error. {e}", "error"); return None
    except (KeyError, ValueError) as e: app.logger.error(f"[Processing] Data mismatch error: {e}.", exc_info=True); flash(f"Error: Data structure mismatch. {e}", "error"); return None
    except Exception as e: app.logger.error(f"[Processing] Unexpected Error: {e}", exc_info=True); flash(f"Error during data processing: {e}", "error"); return None
    finally:
        # --- Ensure processing connection is closed ---
        if cursor_proc: cursor_proc.close()
        if db_conn_proc and db_conn_proc.is_connected():
            db_conn_proc.close()
            app.logger.info("[Processing] Closed direct DB connection.")

# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    # Use the specific error value defined during loading
    if INDIAN_STATES == ["Error: State Data Load Failed"]:
        flash("Application Error: Could not load state data.", "error")
        # Pass an error flag or specific state list to the template
        return render_template('index.html', states=INDIAN_STATES, form_data={}, error_state=True)

    if request.method == 'POST':
        user_name = request.form.get('name', '').strip()
        user_state = request.form.get('user_state')
        user_age = request.form.get('age', '').strip() # Get age, strip whitespace
        user_sex = request.form.get('sex', '') # Get sex

        errors = False
        if not user_name:
            flash('Name is required.', 'error')
            errors = True
        if not user_state or user_state not in INDIAN_STATES:
            flash('Please select a valid state.', 'error')
            errors = True
        # *** ADDED AGE VALIDATION ***
        if not user_age:
            flash('Age is required.', 'error')
            errors = True
        elif not user_age.isdigit() or int(user_age) <= 0: # Optional: Add stronger validation
             flash('Please enter a valid positive number for age.', 'error')
             errors = True
        # Check for sex (radio buttons with 'required' usually handle this client-side,
        # but backend check is safer)
        if not user_sex:
             flash('Sex selection is required.', 'error')
             errors = True


        if errors:
            # Pass submitted data back to repopulate the form
            return render_template('index.html', states=INDIAN_STATES, form_data=request.form)

        # If validation passes, prepare user_info dictionary
        # Ensure age is passed correctly
        user_info = {'name': user_name, 'state': user_state, 'age': user_age, 'sex': user_sex}
        app.logger.info(f"Index POST: User '{user_name}', State: '{user_state}', Age: '{user_age}', Sex: '{user_sex}'. Redirecting to quiz.")
        return redirect(url_for('quiz', **user_info))

    # GET request: render the form
    return render_template('index.html', states=INDIAN_STATES, form_data={})

@app.route('/quiz')
def quiz():
    user_info = {k: request.args.get(k) for k in ['name', 'state', 'age', 'sex']}
    # Add check for age as well, since it's now mandatory
    if not user_info['name'] or not user_info['state'] or not user_info['age']:
        app.logger.warning(f"Quiz access missing required info: Name='{user_info['name']}', State='{user_info['state']}', Age='{user_info['age']}'.")
        flash('User information (Name, State, Age) is required.', 'error')
        return redirect(url_for('index'))
    if INDIAN_STATES == ["Error: State Data Load Failed"] or not ALL_STEREOTYPE_DATA:
        app.logger.error("Quiz cannot display: Stereotype/State data failed load.")
        flash('Application Error: Stereotype data missing or invalid.', 'error')
        return redirect(url_for('index'))

    filtered_quiz_items = [item for item in ALL_STEREOTYPE_DATA if item['state'] == user_info['state']]
    app.logger.info(f"Quiz page: User '{user_info['name']}', State '{user_info['state']}'. Items: {len(filtered_quiz_items)}.")
    if not filtered_quiz_items:
        flash(f"No stereotypes listed for {user_info['state']}.", 'info')
    return render_template('quiz.html', quiz_items=filtered_quiz_items, user_info=user_info)

@app.route('/submit', methods=['POST'])
def submit():
    app.logger.info("Submit route entered.")
    cursor = get_db()
    if not cursor:
        app.logger.error("Submit failed: Could not get DB cursor."); return redirect(url_for('index'))
    db_connection = g.db
    app.logger.debug("Submit: DB cursor obtained.")
    try:
        user_name = request.form.get('user_name','').strip()
        user_state = request.form.get('user_state')
        user_age_str = request.form.get('user_age') # Still retrieve as string first
        user_sex = request.form.get('user_sex') or None # Default to None if not provided

        # Add validation here too, as a safeguard against direct POSTs or URL manipulation
        if not user_name or not user_state or not user_age_str or not user_sex:
            app.logger.warning(f"Submit rejected: Missing user info. Name='{user_name}', State='{user_state}', Age='{user_age_str}', Sex='{user_sex}'")
            flash("User information missing (Name, State, Age, Sex).", 'error')
            return redirect(url_for('index'))

        # Convert age to integer, handle potential errors (though should be valid now)
        try:
            user_age = int(user_age_str)
            if user_age <= 0: raise ValueError("Age must be positive")
        except (ValueError, TypeError):
            user_age = None # Should not happen if index validation works
            app.logger.warning(f"Submit: Invalid age '{user_age_str}' received despite validation. Setting to None.")
            # Optionally, redirect with error:
            # flash("Invalid age submitted.", 'error')
            # return redirect(url_for('index'))


        results_to_insert = []
        processed_indices = set()
        app.logger.debug("Submit: Parsing form data...")
        for key in request.form:
            if key.startswith('annotation_'):
                identifier = key.replace('annotation_', '')
                if identifier in processed_indices: continue
                processed_indices.add(identifier)
                superset = request.form.get(f'superset_{identifier}')
                category = request.form.get(f'category_{identifier}')
                annotation = request.form.get(key)
                if not all([superset, category, annotation]):
                    app.logger.warning(f"Submit skipping item '{identifier}': missing data."); continue

                offensiveness = -1 # Default: Not applicable or not rated
                if annotation == 'Stereotype':
                    rating_str = request.form.get(f'offensiveness_{identifier}')
                    if rating_str is not None and rating_str.isdigit():
                        rating_val = int(rating_str)
                        if 0 <= rating_val <= 5:
                            offensiveness = rating_val
                        else:
                            app.logger.warning(f"Submit out-of-range rating ({rating_val}) for '{identifier}'. Setting to -1.")
                    elif rating_str is not None:
                         app.logger.warning(f"Submit non-integer rating ('{rating_str}') for '{identifier}'. Setting to -1.")
                    # else: No rating provided for a stereotype, offensiveness remains -1

                results_to_insert.append({
                    'user_name': user_name,
                    'user_state': user_state,
                    'user_age': user_age, # Use the validated integer age
                    'user_sex': user_sex,
                    'category': category,
                    'attribute_superset': superset,
                    'annotation': annotation,
                    'offensiveness_rating': offensiveness
                })

        app.logger.info(f"Submit: Prepared {len(results_to_insert)} items for insertion for user '{user_name}'.")
        if results_to_insert:
            sql = f"""INSERT INTO {RESULTS_TABLE}
                      (user_name, user_state, user_age, user_sex, category, attribute_superset, annotation, offensiveness_rating)
                      VALUES (%(user_name)s, %(user_state)s, %(user_age)s, %(user_sex)s, %(category)s, %(attribute_superset)s, %(annotation)s, %(offensiveness_rating)s)"""
            try:
                app.logger.info(f"Submit: Executing INSERT for {len(results_to_insert)} rows...")
                # executemany expects a list of dictionaries
                cursor.executemany(sql, results_to_insert)
                rowcount = cursor.rowcount
                app.logger.info(f"Submit: executemany successful, rowcount={rowcount}. Committing...")
                db_connection.commit()
                app.logger.info(f"Submit: COMMIT successful. Inserted {rowcount} results for user '{user_name}'.")
                flash(f"Successfully submitted {len(results_to_insert)} responses. Thank you!", 'success')
            except MySQLError as db_err:
                app.logger.error(f"Submit DB Insert Error user '{user_name}': {db_err}", exc_info=True)
                try: db_connection.rollback(); app.logger.info("Submit: Rolled back on insert error.")
                except Exception as rb_err: app.logger.error(f"Submit: Rollback failed after insert error: {rb_err}")
                flash("Error saving responses. Please try again.", 'error'); return redirect(url_for('index')) # Redirect to index on error
            except Exception as e:
                app.logger.error(f"Submit Unexpected DB Insert Error user '{user_name}': {e}", exc_info=True)
                try: db_connection.rollback(); app.logger.info("Submit: Rolled back on unexpected insert error.")
                except Exception as rb_err: app.logger.error(f"Submit: Rollback failed after unexpected insert error: {rb_err}")
                flash("Unexpected error saving data. Please try again.", 'error'); return redirect(url_for('index')) # Redirect to index on error
        else:
             app.logger.warning(f"Submit: No valid results parsed for user '{user_name}'."); flash("No valid responses found to submit.", 'warning')

        app.logger.info(f"Submit: Redirecting user '{user_name}' to thank_you page."); return redirect(url_for('thank_you'))
    except Exception as e:
        app.logger.error(f"Submit: Unexpected error in route logic: {e}", exc_info=True); flash("Unexpected error during submission.", 'error')
        try:
            if 'db' in g and g.db and g.db.is_connected(): g.db.rollback(); app.logger.info("Submit: Rolled back in final exception handler.")
        except Exception as rb_err: app.logger.error(f"Submit: Rollback failed in final exception handler: {rb_err}")
        return redirect(url_for('index')) # Redirect to index on error

@app.route('/thank_you')
def thank_you(): return render_template('thank_you.html')

# --- Admin Routes ---
# admin_view() remains the same
@app.route('/admin')
def admin_view():
    app.logger.info("Admin route entered.")
    app.logger.warning("Accessed unsecured /admin route.")
    cursor = get_db()
    if not cursor: app.logger.error("Admin view failed: No DB cursor."); return redirect(url_for('index'))
    app.logger.debug("Admin view: DB cursor obtained.")
    results_data = []
    try:
        sql_query = f"SELECT * FROM {RESULTS_TABLE} ORDER BY timestamp DESC"
        app.logger.info(f"Admin view: Executing query: {sql_query}")
        cursor.execute(sql_query)
        app.logger.debug("Admin view: Query execution complete.")
        results_data = cursor.fetchall()
        app.logger.info(f"Admin view: Fetched {len(results_data)} results.")
    except MySQLError as err:
        app.logger.error(f"Admin view: Error executing SELECT: {err}", exc_info=True); flash('Error fetching results data.', 'error'); return render_template('admin.html', results=[])
    except Exception as e:
         app.logger.error(f"Admin view: Unexpected error fetching data: {e}", exc_info=True); flash('Unexpected error loading admin data.', 'error'); return render_template('admin.html', results=[])
    app.logger.debug("Admin view: Rendering template.")
    return render_template('admin.html', results=results_data)

# download_processed_data() uses the refactored generate_aggregated_data()
@app.route('/admin/download_processed')
def download_processed_data():
    app.logger.warning("Accessed unsecured /admin/download_processed route.")
    aggregated_df = generate_aggregated_data() # Use refactored data fetching
    if aggregated_df is None: app.logger.error("Download processed: Aggregation failed."); return redirect(url_for('admin_view'))
    if aggregated_df.empty: app.logger.warning("Download processed: Aggregation returned empty."); flash("No data to process/download.", "warning"); return redirect(url_for('admin_view'))
    try:
        buffer = io.BytesIO()
        # Ensure UTF-8 BOM for Excel compatibility if needed, otherwise standard UTF-8 is fine
        # buffer.write(u'\ufeff'.encode('utf8')) # Optional BOM
        buffer.write(aggregated_df.to_csv(index=False, encoding='utf-8').encode('utf-8'))
        buffer.seek(0)
        app.logger.info("Sending aggregated CSV file download.")
        return send_file(buffer, mimetype='text/csv; charset=utf-8', download_name='final_aggregated_stereotypes.csv', as_attachment=True)
    except Exception as e: app.logger.error(f"Download processed: Error generating CSV: {e}", exc_info=True); flash(f"Error creating download file: {e}", "error"); return redirect(url_for('admin_view'))


# --- REFACTORED download_raw_data ---
@app.route('/admin/download_raw')
def download_raw_data():
    app.logger.warning("Accessed unsecured /admin/download_raw route.")
    db_conn_raw = None
    cursor_raw = None
    raw_results_list = []

    try:
        # --- Fetch raw data using direct connection ---
        db_host = current_app.config['MYSQL_HOST']
        db_user = current_app.config['MYSQL_USER']
        db_password = current_app.config['MYSQL_PASSWORD']
        db_name = current_app.config['MYSQL_DB']
        db_port = current_app.config['MYSQL_PORT']

        if not all([db_host, db_user, db_password, db_name]):
            app.logger.error("[Raw Download] Aborted: DB config missing.")
            flash("Server configuration error.", "error")
            return redirect(url_for('admin_view'))

        app.logger.info(f"[Raw Download] Attempting direct DB connection: User={db_user}, Host={db_host}, Port={db_port}, DB={db_name}")
        db_conn_raw = mysql.connector.connect(
            host=db_host, user=db_user, password=db_password, database=db_name, port=db_port, connection_timeout=10
        )
        cursor_raw = db_conn_raw.cursor(dictionary=True) # Fetch as dictionaries

        app.logger.info("[Raw Download] Fetching data via direct cursor...")
        cursor_raw.execute(f"SELECT * FROM {RESULTS_TABLE} ORDER BY timestamp DESC")
        raw_results_list = cursor_raw.fetchall()
        app.logger.info(f"[Raw Download] Fetched {len(raw_results_list)} rows.")

        if not raw_results_list:
            flash("Raw results table is empty.", "warning")
            return redirect(url_for('admin_view'))

        # --- Convert list of dicts to DataFrame ---
        raw_results_df = pd.DataFrame(raw_results_list)
        app.logger.info("[Raw Download] Converted fetched results to DataFrame.")

        # --- Generate CSV from DataFrame ---
        buffer = io.BytesIO()
        # buffer.write(u'\ufeff'.encode('utf8')) # Optional BOM
        buffer.write(raw_results_df.to_csv(index=False, encoding='utf-8').encode('utf-8'))
        buffer.seek(0)
        app.logger.info("[Raw Download] Sending raw CSV file download.")
        return send_file(buffer, mimetype='text/csv; charset=utf-8', download_name='raw_quiz_results.csv', as_attachment=True)

    except MySQLError as e:
        app.logger.error(f"[Raw Download] DB Error during direct fetch: {e}", exc_info=True)
        flash(f"Error fetching raw data: {e}", "error")
        return redirect(url_for('admin_view'))
    except Exception as e:
        app.logger.error(f"[Raw Download] Unexpected Error: {e}", exc_info=True)
        flash(f"Error preparing raw data download: {e}", "error")
        return redirect(url_for('admin_view'))
    finally:
        # --- Ensure raw download connection is closed ---
        if cursor_raw: cursor_raw.close()
        if db_conn_raw and db_conn_raw.is_connected():
            db_conn_raw.close()
            app.logger.info("[Raw Download] Closed direct DB connection.")


# --- Main Execution (for Local Development ONLY) ---
if __name__ == '__main__':
    app.logger.info("Starting Flask development server (DO NOT USE IN PRODUCTION)...")
    # Initialize DB Table on startup if running directly
    with app.app_context():
        if not init_db_table():
             app.logger.error("Failed to initialize database table during startup. Check DB connection and schema.")
             # Decide if you want to proceed or exit
             # sys.exit("Database initialization failed.")
        if INDIAN_STATES == ["Error: State Data Load Failed"]:
            app.logger.error("Failed to load state data during startup. Check CSV file and path.")
            # Decide if you want to proceed or exit
            # sys.exit("State data loading failed.")

    host = os.environ.get('FLASK_RUN_HOST', '127.0.0.1')
    try: port = int(os.environ.get('FLASK_RUN_PORT', '5000'))
    except ValueError: port = 5000
    debug_mode = os.environ.get('FLASK_ENV') == 'development' or os.environ.get('FLASK_DEBUG') == '1'
    if not debug_mode: app.logger.warning("FLASK_ENV not 'development'. Running dev server without debug.")
    app.run(host=host, port=port, debug=debug_mode)