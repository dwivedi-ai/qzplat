# stereotype_quiz_app/app.py

import os
import csv
import io # Needed for in-memory CSV generation
import mysql.connector # Use MySQL connector
from mysql.connector import Error as MySQLError # Import specific error class
from flask import Flask, render_template, request, redirect, url_for, g, flash, Response, send_file # Added Response, send_file
import pandas as pd # Needed for processing & raw download
import numpy as np  # Needed for processing

# --- Configuration ---

# Path for CSV (relative to app.py)
CSV_FILE_PATH = os.path.join('data', 'stereotypes.csv')
# Path for MySQL Schema file (relative to app.py) - Make sure this file uses MySQL syntax
SCHEMA_FILE = 'schema_mysql.sql' # Assuming you created schema_mysql.sql

# Flask Secret Key (Use a strong, unique key)
SECRET_KEY = 'respai' # As provided by user

# --- MySQL Configuration (Using User-Provided Credentials) ---
MYSQL_HOST = 'localhost'
MYSQL_USER = 'stereotype_user'
MYSQL_PASSWORD = 'RespAI@2025' # As provided by user
MYSQL_DB = 'stereotype_quiz_db'
MYSQL_PORT = 3306
RESULTS_TABLE = 'results' # Name of the table in the database

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['MYSQL_HOST'] = MYSQL_HOST
app.config['MYSQL_USER'] = MYSQL_USER
app.config['MYSQL_PASSWORD'] = MYSQL_PASSWORD
app.config['MYSQL_DB'] = MYSQL_DB
app.config['MYSQL_PORT'] = MYSQL_PORT

# --- Database Functions (MODIFIED FOR MYSQL) ---

def get_db():
    """Opens a new MySQL database connection and cursor if none exist for the current request context."""
    if 'db' not in g:
        try:
            g.db = mysql.connector.connect(
                host=app.config['MYSQL_HOST'],
                user=app.config['MYSQL_USER'],
                password=app.config['MYSQL_PASSWORD'],
                database=app.config['MYSQL_DB'],
                port=app.config['MYSQL_PORT']
            )
            g.cursor = g.db.cursor(dictionary=True) # Dictionary cursor for dict-like rows
            # print("MySQL connection established.") # Less verbose logging
        except MySQLError as err:
            print(f"Error connecting to MySQL: {err}")
            flash('Database connection error. Please try again later or contact admin.', 'error')
            g.db = None
            g.cursor = None
    return getattr(g, 'cursor', None)

@app.teardown_appcontext
def close_db(error):
    """Closes the database cursor and connection at the end of the request."""
    cursor = g.pop('cursor', None)
    if cursor: cursor.close()
    db = g.pop('db', None)
    if db: db.close()
    if error: print(f"App context teardown error: {error}")

def init_db():
    """Connects to MySQL, creates the database if needed, and creates the table using schema_mysql.sql."""
    temp_conn = None
    temp_cursor = None
    try:
        temp_conn = mysql.connector.connect(
            host=app.config['MYSQL_HOST'],
            user=app.config['MYSQL_USER'],
            password=app.config['MYSQL_PASSWORD'],
            port=app.config['MYSQL_PORT']
        )
        temp_cursor = temp_conn.cursor()
        db_name = app.config['MYSQL_DB']
        try:
            temp_cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            temp_conn.commit()
            print(f"Database '{db_name}' checked/created.")
            temp_cursor.execute(f"USE `{db_name}`")
        except MySQLError as err:
            print(f"Failed to create/use database '{db_name}': {err}"); return

        temp_cursor.execute(f"SHOW TABLES LIKE '{RESULTS_TABLE}'")
        if not temp_cursor.fetchone():
            print(f"Creating database table '{RESULTS_TABLE}' using {SCHEMA_FILE}...")
            schema_path = os.path.join(app.root_path, SCHEMA_FILE)
            if not os.path.exists(schema_path):
                # Try alternate path assuming schema_mysql.sql if schema.sql wasn't specified
                schema_path = os.path.join(app.root_path, 'schema_mysql.sql')

            try:
                with open(schema_path, mode='r', encoding='utf-8') as f: sql_script = f.read()
                # Handle potential multiple statements separated by ';'
                for result in temp_cursor.execute(sql_script, multi=True):
                     pass # Consume results if any are returned
                temp_conn.commit()
                print(f"Database table '{RESULTS_TABLE}' created.")
            except FileNotFoundError: print(f"Error: Schema file ({SCHEMA_FILE} or schema_mysql.sql) not found at {schema_path}.")
            except MySQLError as err: print(f"Error executing schema: {err}"); temp_conn.rollback()
            except Exception as e: print(f"Error initializing schema: {e}"); temp_conn.rollback()
        else:
            print(f"Database table '{RESULTS_TABLE}' already exists.")
    except MySQLError as err: print(f"Error during DB initialization: {err}")
    except Exception as e: print(f"Unexpected error during DB init: {e}")
    finally:
        if temp_cursor: temp_cursor.close()
        if temp_conn and temp_conn.is_connected(): temp_conn.close()

# --- Data Loading Function (Corrected Path) ---
def load_stereotype_data(relative_filepath=CSV_FILE_PATH):
    """Loads stereotype data from the CSV within the 'data' directory."""
    stereotype_data = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_filepath = os.path.join(base_dir, relative_filepath)
    print(f"Attempting to load stereotype data from: {full_filepath}")
    try:
        if not os.path.exists(full_filepath): raise FileNotFoundError(f"File not found: {full_filepath}")
        with open(full_filepath, mode='r', encoding='utf-8-sig') as infile: # Use utf-8-sig to handle potential BOM
            reader = csv.DictReader(infile)
            required_cols = ['State', 'Category', 'Superset', 'Subsets']
            if not reader.fieldnames or not all(field in reader.fieldnames for field in required_cols):
                 missing = [c for c in required_cols if c not in (reader.fieldnames or [])]
                 raise ValueError(f"CSV missing required columns: {missing}. Found: {reader.fieldnames}")
            for i, row in enumerate(reader):
                try:
                    state = row.get('State','').strip(); category = row.get('Category','Uncategorized').strip() # Default category
                    superset = row.get('Superset','').strip(); subsets_str = row.get('Subsets','')
                    if not state or not superset: continue # Skip if essential info missing
                    subsets = sorted([s.strip() for s in subsets_str.split(',') if s.strip()]) # Process subsets
                    stereotype_data.append({'state': state, 'category': category, 'superset': superset, 'subsets': subsets})
                except Exception as row_err: print(f"Error processing CSV row {i+1}: {row_err}"); continue
        print(f"Successfully loaded {len(stereotype_data)} stereotype entries from {full_filepath}")
        return stereotype_data
    except FileNotFoundError: print(f"FATAL ERROR: CSV file not found at {full_filepath}."); return []
    except ValueError as ve: print(f"FATAL ERROR processing CSV: {ve}"); return []
    except Exception as e: import traceback; print(f"FATAL ERROR loading data: {e}\n{traceback.format_exc()}"); return []

# --- Load Data & States ---
ALL_STEREOTYPE_DATA = load_stereotype_data()
INDIAN_STATES = sorted(list(set(item['state'] for item in ALL_STEREOTYPE_DATA)))
if not ALL_STEREOTYPE_DATA or not INDIAN_STATES:
    print("\nCRITICAL ERROR: Stereotype data loading failed! Check logs.\n")
    INDIAN_STATES = ["Error: Check Logs"]
print(f"States available for selection: {INDIAN_STATES if INDIAN_STATES != ['Error: Check Logs'] else 'LOADING FAILED'}")


# --- Data Processing Logic (from process_results.py) ---
def calculate_mean_offensiveness(series):
    """Helper: Calculates mean of non-negative ratings, returns NaN if none exist."""
    valid_ratings = series[series >= 0]
    return valid_ratings.mean() if not valid_ratings.empty else np.nan

def generate_aggregated_data():
    """
    Loads raw results, loads definitions, expands annotations, aggregates,
    and returns the final DataFrame. Returns None on critical error.
    """
    print("--- [Processing] Starting data aggregation ---")
    db_conn_proc = None
    try:
        print("[Processing] Connecting to DB to fetch raw results...")
        db_conn_proc = mysql.connector.connect(
            host=app.config['MYSQL_HOST'], user=app.config['MYSQL_USER'],
            password=app.config['MYSQL_PASSWORD'], database=app.config['MYSQL_DB'],
            port=app.config['MYSQL_PORT']
        )
        if not db_conn_proc.is_connected(): raise MySQLError("Processing: Failed to connect to MySQL.")

        results_df = pd.read_sql_query(f"SELECT * FROM {RESULTS_TABLE}", db_conn_proc)
        print(f"[Processing] Loaded {len(results_df)} raw results.")
        if results_df.empty: return pd.DataFrame() # Return empty DataFrame

        results_df['Stereotype_State'] = results_df['user_state']

        stereotypes_path = os.path.join(app.root_path, CSV_FILE_PATH)
        if not os.path.exists(stereotypes_path): raise FileNotFoundError(f"Processing: Stereotypes file not found at {stereotypes_path}")
        print(f"[Processing] Loading definitions from: {stereotypes_path}")
        stereotypes_df = pd.read_csv(stereotypes_path, encoding='utf-8-sig')
        stereotypes_df['Subsets_List'] = stereotypes_df['Subsets'].fillna('').astype(str).apply(lambda x: [s.strip() for s in x.split(',') if s.strip()])
        print(f"[Processing] Loaded {len(stereotypes_df)} definitions.")
        subset_lookup = stereotypes_df.set_index(['State', 'Category', 'Superset'])['Subsets_List'].to_dict()

        print("[Processing] Expanding annotations...")
        expanded_rows = []
        for index, result_row in results_df.iterrows():
            state = result_row['Stereotype_State']; category = result_row['category']
            superset = result_row['attribute_superset']; annotation = result_row['annotation']
            rating = result_row['offensiveness_rating']
            expanded_rows.append({'Stereotype_State': state, 'Category': category, 'Attribute': superset, 'annotation': annotation, 'offensiveness_rating': rating})
            subsets_list = subset_lookup.get((state, category, superset), [])
            for subset in subsets_list:
                expanded_rows.append({'Stereotype_State': state, 'Category': category, 'Attribute': subset, 'annotation': annotation, 'offensiveness_rating': rating})

        if not expanded_rows: return pd.DataFrame() # Return empty DataFrame
        expanded_annotations_df = pd.DataFrame(expanded_rows)
        print(f"[Processing] Created {len(expanded_annotations_df)} expanded rows.")

        print("[Processing] Aggregating results...")
        grouped = expanded_annotations_df.groupby(['Stereotype_State', 'Category', 'Attribute'])
        aggregated_data = grouped.agg(
            Stereotype_Votes=('annotation', lambda x: (x == 'Stereotype').sum()),
            Not_Stereotype_Votes=('annotation', lambda x: (x == 'Not a Stereotype').sum()),
            Not_Sure_Votes=('annotation', lambda x: (x == 'Not sure').sum()),
            Average_Offensiveness=('offensiveness_rating', calculate_mean_offensiveness)
        ).reset_index()
        aggregated_data['Average_Offensiveness'] = aggregated_data['Average_Offensiveness'].round(2)
        print(f"[Processing] Aggregation complete. Result has {len(aggregated_data)} rows.")
        print("--- [Processing] Finished data aggregation successfully ---")
        return aggregated_data

    except FileNotFoundError as e: print(f"[Processing] Error: {e}"); flash(f"Error: Input file not found. {e}", "error"); return None
    except MySQLError as e: print(f"[Processing] Database Error: {e}"); flash(f"Error: Database error. {e}", "error"); return None
    except KeyError as e: print(f"[Processing] Error: Missing column: {e}."); flash(f"Error: Data structure mismatch (Column: {e}).", "error"); return None
    except Exception as e: import traceback; print(f"[Processing] Unexpected Error:\n{traceback.format_exc()}"); flash(f"Error during data processing: {e}", "error"); return None
    finally:
        if db_conn_proc and db_conn_proc.is_connected(): db_conn_proc.close(); print("[Processing] Closed DB connection.")


# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    """Handles the initial user info form page. Name is mandatory."""
    if request.method == 'POST':
        user_name = request.form.get('name', '').strip()
        user_state = request.form.get('user_state')
        # Validation
        errors = False
        if not user_name: flash('Name is required.', 'error'); errors = True
        if not user_state or user_state not in INDIAN_STATES: flash('Please select a valid state.', 'error'); errors = True
        if errors: return render_template('index.html', states=INDIAN_STATES, form_data=request.form)
        # Prepare user info for redirect
        user_info = {'name': user_name, 'state': user_state, 'age': request.form.get('age',''), 'sex': request.form.get('sex','')}
        print(f"Index POST successful. Redirecting to quiz with: {user_info}")
        return redirect(url_for('quiz', **user_info))
    # GET request
    return render_template('index.html', states=INDIAN_STATES, form_data={})


@app.route('/quiz')
def quiz():
    """Displays the quiz questions FILTERED by the user's state."""
    user_info = {'name': request.args.get('name'), 'state': request.args.get('state'), 'age': request.args.get('age'), 'sex': request.args.get('sex')}
    # Validate essential info
    if not user_info['name'] or not user_info['state']:
         print("Redirecting to index: User name or state missing in quiz URL.")
         flash('User info missing. Please start again.', 'error'); return redirect(url_for('index'))
    # Check data loaded OK
    if not ALL_STEREOTYPE_DATA: flash('Error loading stereotype data.', 'error'); return redirect(url_for('index'))
    # Filter items
    filtered_quiz_items = [item for item in ALL_STEREOTYPE_DATA if item['state'] == user_info['state']]
    print(f"Displaying quiz for user '{user_info['name']}', state '{user_info['state']}'. Found {len(filtered_quiz_items)} items.")
    if not filtered_quiz_items: flash(f"No stereotypes found for {user_info['state']}.", 'info')
    # Render template
    return render_template('quiz.html', quiz_items=filtered_quiz_items, user_info=user_info)


@app.route('/submit', methods=['POST'])
def submit():
    """Handles the submission of the quiz answers to MySQL."""
    cursor = get_db();
    if not cursor: return redirect(url_for('index')) # get_db flashes error
    db_connection = g.db
    try:
        # Retrieve user info from hidden fields
        user_name = request.form.get('user_name'); user_state = request.form.get('user_state')
        user_age_str = request.form.get('user_age'); user_sex = request.form.get('user_sex') or None
        if not user_name or not user_state: flash("User info missing.", 'error'); return redirect(url_for('index'))
        try: user_age = int(user_age_str) if user_age_str else None
        except: user_age = None # Handle conversion error gracefully

        results_to_insert = []
        processed_indices = set()
        # Iterate through form to find annotations
        for key in request.form:
            if key.startswith('annotation_'):
                identifier = key.replace('annotation_', '')
                if identifier in processed_indices: continue
                processed_indices.add(identifier)

                superset = request.form.get(f'superset_{identifier}')
                category = request.form.get(f'category_{identifier}')
                annotation = request.form.get(key)
                # Skip if essential parts missing for this item
                if not all([superset, category, annotation]): continue

                offensiveness = -1 # Default value (Required NOT NULL in DB)
                if annotation == 'Stereotype':
                    rating_str = request.form.get(f'offensiveness_{identifier}')
                    try:
                        if rating_str is not None:
                            offensiveness = int(rating_str)
                            if not (0 <= offensiveness <= 5): offensiveness = -1 # Enforce range
                        # else: offensiveness remains -1 (if radio not selected)
                    except (ValueError, TypeError): offensiveness = -1 # Handle non-integer

                results_to_insert.append({
                    'user_name': user_name, 'user_state': user_state, 'user_age': user_age,
                    'user_sex': user_sex, 'category': category, 'attribute_superset': superset,
                    'annotation': annotation, 'offensiveness_rating': offensiveness
                })

        # Insert collected results if any
        if results_to_insert:
            sql = f"INSERT INTO {RESULTS_TABLE} (user_name, user_state, user_age, user_sex, category, attribute_superset, annotation, offensiveness_rating) VALUES (%(user_name)s, %(user_state)s, %(user_age)s, %(user_sex)s, %(category)s, %(attribute_superset)s, %(annotation)s, %(offensiveness_rating)s)"
            try:
                cursor.executemany(sql, results_to_insert)
                db_connection.commit()
                print(f"Inserted {cursor.rowcount} results into the database for user '{user_name}'.")
                flash(f"Successfully submitted {len(results_to_insert)} responses. Thank you!", 'success')
            except MySQLError as db_err:
                 print(f"DB Insert Error: {db_err}"); db_connection.rollback(); flash("Error saving responses.", 'error'); return redirect(url_for('index'))
            except Exception as e:
                 import traceback; print(f"Unexpected DB Insert Error: {e}\n{traceback.format_exc()}");
                 try: db_connection.rollback()
                 except: pass
                 flash("Unexpected error saving data.", 'error'); return redirect(url_for('index'))
        else:
             print("Warning: No valid results parsed from submission.")
             flash("No valid responses were found in your submission.", 'warning')

        return redirect(url_for('thank_you'))

    except Exception as e: # Broad catch for unexpected errors in route logic
        import traceback; print(f"Submit Route Error: {e}\n{traceback.format_exc()}"); flash("An unexpected error occurred during submission.", 'error'); return redirect(url_for('index'))


@app.route('/thank_you')
def thank_you():
    """Displays the thank you page."""
    return render_template('thank_you.html')


# --- Admin Routes ---
@app.route('/admin')
def admin_view():
    """Displays the collected results from the MySQL database."""
    # !! SECURITY WARNING: Add authentication here before deployment !!
    cursor = get_db()
    if not cursor: return redirect(url_for('index')) # Error flashed in get_db
    results_data = []
    try:
        cursor.execute(f"SELECT * FROM {RESULTS_TABLE} ORDER BY timestamp DESC")
        results_data = cursor.fetchall()
        print(f"Admin view: Fetched {len(results_data)} results.")
    except MySQLError as err:
        print(f"Error fetching admin data: {err}")
        flash('Error fetching results data.', 'error')
    except Exception as e:
         print(f"Unexpected error fetching admin data: {e}")
         flash('Unexpected error loading admin data.', 'error')
    return render_template('admin.html', results=results_data)

@app.route('/admin/download_processed')
def download_processed_data():
    """Triggers data processing and sends aggregated results as CSV."""
    # !! SECURITY WARNING: Add authentication here before deployment !!
    print("Download request received for processed data.")
    aggregated_df = generate_aggregated_data() # Call the processing function

    if aggregated_df is None: # Check if processing function indicated an error
        print("Processing failed. Redirecting back to admin.")
        # Flash message should already be set by generate_aggregated_data
        return redirect(url_for('admin_view'))
    if aggregated_df.empty:
        flash("No data to process or results table is empty.", "warning")
        print("Processing returned empty DataFrame. Redirecting back to admin.")
        return redirect(url_for('admin_view'))

    # Generate CSV in memory
    try:
        print("Processing successful. Generating CSV in memory...")
        buffer = io.BytesIO()
        aggregated_df.to_csv(buffer, index=False, encoding='utf-8')
        buffer.seek(0)
        print("Sending aggregated CSV file for download...")
        return send_file( buffer, mimetype='text/csv', download_name='final_aggregated_stereotypes.csv', as_attachment=True )
    except Exception as e:
        import traceback; print(f"Error generating/sending processed CSV:\n{traceback.format_exc()}"); flash(f"Error creating download file: {e}", "error"); return redirect(url_for('admin_view'))

@app.route('/admin/download_raw')
def download_raw_data():
    """Fetches all raw results and sends them as CSV."""
    # !! SECURITY WARNING: Add authentication here before deployment !!
    print("Raw data download request received.")
    db_conn_raw = None
    try:
        print("[Raw Download] Connecting to DB...")
        db_conn_raw = mysql.connector.connect(
            host=app.config['MYSQL_HOST'], user=app.config['MYSQL_USER'],
            password=app.config['MYSQL_PASSWORD'], database=app.config['MYSQL_DB'],
            port=app.config['MYSQL_PORT']
        )
        if not db_conn_raw.is_connected(): raise MySQLError("Raw Download: Failed to connect.")

        print("[Raw Download] Fetching data...")
        raw_results_df = pd.read_sql_query(f"SELECT * FROM {RESULTS_TABLE} ORDER BY timestamp DESC", db_conn_raw)
        print(f"[Raw Download] Fetched {len(raw_results_df)} raw rows.")

        if raw_results_df.empty: flash("Raw results table is empty.", "warning"); return redirect(url_for('admin_view'))

        print("[Raw Download] Generating CSV in memory...")
        buffer = io.BytesIO()
        raw_results_df.to_csv(buffer, index=False, encoding='utf-8')
        buffer.seek(0)
        print("[Raw Download] Sending raw CSV file...")
        return send_file( buffer, mimetype='text/csv', download_name='raw_quiz_results.csv', as_attachment=True )

    except (MySQLError, pd.errors.DatabaseError) as e: print(f"[Raw Download] DB/Pandas Error: {e}"); flash(f"Error fetching/reading raw data: {e}", "error"); return redirect(url_for('admin_view'))
    except Exception as e: import traceback; print(f"[Raw Download] Unexpected Error:\n{traceback.format_exc()}"); flash(f"Error preparing raw data download: {e}", "error"); return redirect(url_for('admin_view'))
    finally:
        if db_conn_raw and db_conn_raw.is_connected(): db_conn_raw.close(); print("[Raw Download] Closed DB connection.")


# --- Main Execution ---
if __name__ == '__main__':
    # Ensure DB and table exist on startup
    print("Initializing database...")
    with app.app_context(): # Ensure init_db runs within app context if needed
        init_db()
    print("Starting Flask application...")
    # Set debug=False for production environments
    app.run(debug=True, host='0.0.0.0', port=5000)