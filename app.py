# stereotype_quiz_app/app.py

import os
import csv
import io # Needed for in-memory CSV generation
import mysql.connector # Use MySQL connector
from mysql.connector import Error as MySQLError # Import specific error class
from flask import Flask, render_template, request, redirect, url_for, g, flash, Response, send_file
import pandas as pd # Needed for processing & raw download
import numpy as np  # Needed for processing
import traceback # For detailed error logging

# --- Configuration ---

# Path for CSV (relative to app.py's location)
CSV_FILE_PATH = os.path.join('data', 'stereotypes.csv')
# Path for MySQL Schema file (relative to app.py's location)
SCHEMA_FILE = 'schema_mysql.sql'

# --- Configuration from Environment Variables (Railway compatible names) ---

# Flask Secret Key (Read from environment variable)
# IMPORTANT: Set this variable in your Railway service settings!
# Provide a default ONLY for local development if .env is not used/loaded
SECRET_KEY = os.environ.get('SECRET_KEY', 'a_very_weak_default_secret_key_change_me_if_local')

# MySQL Configuration - Using environment variable names commonly provided by Railway
# Provide local defaults matching your original setup for convenience if Railway vars aren't present
# NOTE: These defaults will likely NOT work when deployed; Railway's injected variables are essential.
MYSQL_HOST = os.environ.get('MYSQLHOST', 'localhost') # Railway uses MYSQLHOST
MYSQL_USER = os.environ.get('MYSQLUSER', 'stereotype_user') # Railway uses MYSQLUSER
MYSQL_PASSWORD = os.environ.get('MYSQLPASSWORD', 'RespAI@2025') # Railway uses MYSQLPASSWORD (keep local default for convenience)
MYSQL_DB = os.environ.get('MYSQLDATABASE', 'stereotype_quiz_db') # Railway uses MYSQLDATABASE
MYSQL_PORT = int(os.environ.get('MYSQLPORT', 3306)) # Railway uses MYSQLPORT, ensure it's integer

RESULTS_TABLE = 'results' # Name of the table in the database

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY # Use the variable read from env/default

# Store DB config IN app.config for consistent access throughout the app
# This ensures all parts of the app (get_db, init_db, processing) use the same config source.
app.config['MYSQL_HOST'] = MYSQL_HOST
app.config['MYSQL_USER'] = MYSQL_USER
app.config['MYSQL_PASSWORD'] = MYSQL_PASSWORD
app.config['MYSQL_DB'] = MYSQL_DB
app.config['MYSQL_PORT'] = MYSQL_PORT

print("--- Application Configuration ---")
print(f"SECRET_KEY: {'Set (likely from env)' if SECRET_KEY != 'a_very_weak_default_secret_key_change_me_if_local' else 'Using default (UNSAFE FOR PROD!)'}")
print(f"MYSQL_HOST: {app.config['MYSQL_HOST']}")
print(f"MYSQL_USER: {app.config['MYSQL_USER']}")
print(f"MYSQL_PASSWORD: {'Set (likely from env)' if app.config['MYSQL_PASSWORD'] else 'Not Set'}") # Avoid printing password
print(f"MYSQL_DB: {app.config['MYSQL_DB']}")
print(f"MYSQL_PORT: {app.config['MYSQL_PORT']}")
print("-------------------------------")


# --- Database Functions ---

def get_db():
    """Opens a new MySQL database connection and cursor if none exist for the current request context."""
    if 'db' not in g:
        try:
            # Use configuration stored in app.config
            g.db = mysql.connector.connect(
                host=app.config['MYSQL_HOST'],
                user=app.config['MYSQL_USER'],
                password=app.config['MYSQL_PASSWORD'],
                database=app.config['MYSQL_DB'],
                port=app.config['MYSQL_PORT'],
                connection_timeout=10 # Add a connection timeout
            )
            g.cursor = g.db.cursor(dictionary=True) # Dictionary cursor for dict-like rows
            # print("DEBUG: MySQL connection established for this request.") # Optional debug logging
        except MySQLError as err:
            print(f"ERROR connecting to MySQL: {err}")
            print(f"DEBUG: Connection attempted with Host={app.config['MYSQL_HOST']}, User={app.config['MYSQL_USER']}, DB={app.config['MYSQL_DB']}, Port={app.config['MYSQL_PORT']}")
            # Log the error properly in a real app
            flash('Database connection error. Please try again later or contact admin.', 'error')
            g.db = None
            g.cursor = None
            # Optionally raise an exception or handle differently depending on desired app behavior
    # Return the cursor if connection was successful, otherwise return None
    return getattr(g, 'cursor', None)

@app.teardown_appcontext
def close_db(error):
    """Closes the database cursor and connection at the end of the request."""
    cursor = g.pop('cursor', None)
    if cursor:
        try:
            cursor.close()
        except Exception as e:
            print(f"Error closing cursor: {e}") # Log error
    db = g.pop('db', None)
    if db and db.is_connected():
        try:
             db.close()
             # print("DEBUG: MySQL connection closed for this request.") # Optional debug logging
        except Exception as e:
             print(f"Error closing DB connection: {e}") # Log error
    if error:
        print(f"App context teardown error detected: {error}") # Log error


def init_db():
    """Connects to MySQL, creates the database IF NOT EXISTS, and creates the table IF NOT EXISTS using schema_mysql.sql."""
    temp_conn = None
    temp_cursor = None
    # Use app.config consistently
    db_host = app.config['MYSQL_HOST']
    db_user = app.config['MYSQL_USER']
    db_password = app.config['MYSQL_PASSWORD']
    db_port = app.config['MYSQL_PORT']
    db_name = app.config['MYSQL_DB'] # The target database name

    print(f"--- init_db: Attempting DB Initialization (Host: {db_host}, User: {db_user}, DB: {db_name}, Port: {db_port}) ---")

    try:
        # First, connect to the MySQL server *without* specifying a database
        # This is necessary to be able to execute CREATE DATABASE
        print(f"init_db: Connecting to MySQL server ({db_host}:{db_port}) to check/create database...")
        temp_conn = mysql.connector.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            port=db_port,
            connection_timeout=15 # Slightly longer timeout for init
        )
        temp_cursor = temp_conn.cursor()
        print(f"init_db: Connected to MySQL server. Checking/creating database '{db_name}'...")

        # Create the database if it doesn't exist
        try:
            # Use backticks for safety with potentially reserved keywords or special chars in DB name
            temp_cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            temp_conn.commit() # Commit the database creation
            print(f"init_db: Database '{db_name}' checked/created.")
        except MySQLError as err:
            # If CREATE DATABASE fails (e.g., permissions), we likely can't proceed
            print(f"CRITICAL init_db ERROR: Failed to create database '{db_name}': {err}. Check user permissions.")
            return # Exit init_db if database cannot be created

        # Switch to the target database
        try:
            temp_cursor.execute(f"USE `{db_name}`")
            print(f"init_db: Switched to database '{db_name}'.")
        except MySQLError as err:
            # If USE database fails (e.g., DB doesn't exist despite CREATE IF NOT EXISTS, or permissions)
            print(f"CRITICAL init_db ERROR: Failed to switch to database '{db_name}': {err}. Aborting table creation.")
            return # Exit init_db if we can't use the database

        # Now check if the table exists within the selected database
        print(f"init_db: Checking for table '{RESULTS_TABLE}' in database '{db_name}'...")
        temp_cursor.execute(f"SHOW TABLES LIKE '{RESULTS_TABLE}'")
        table_exists = temp_cursor.fetchone()

        if not table_exists:
            print(f"init_db: Table '{RESULTS_TABLE}' not found. Attempting creation using schema file: {SCHEMA_FILE}...")
            schema_path = os.path.join(app.root_path, SCHEMA_FILE)

            # Try alternate common name if specific one not found
            if not os.path.exists(schema_path):
                schema_path_alt = os.path.join(app.root_path, 'schema_mysql.sql')
                if os.path.exists(schema_path_alt):
                    print(f"init_db: Warning: {SCHEMA_FILE} not found, using alternative {schema_path_alt}")
                    schema_path = schema_path_alt
                else:
                    raise FileNotFoundError(f"init_db: Schema file not found at expected locations: {schema_path} or {schema_path_alt}")

            try:
                with open(schema_path, mode='r', encoding='utf-8') as f:
                    sql_script = f.read()

                # Execute the script, handling multiple statements separated by ';'
                print(f"init_db: Executing SQL script from {schema_path}...")
                # Use multi=True for scripts potentially containing multiple statements
                # Iterate through results to ensure all statements are processed
                statement_results = temp_cursor.execute(sql_script, multi=True)
                for i, result in enumerate(statement_results):
                    # Basic logging of statement execution result
                    print(f"  - Statement {i+1}: Rows affected/selected: {result.rowcount if hasattr(result, 'rowcount') else 'N/A'}")
                    if result.with_rows:
                         result.fetchall() # Consume rows if any to allow next statement

                temp_conn.commit() # Commit table creation
                print(f"init_db: Database table '{RESULTS_TABLE}' created successfully.")
            except FileNotFoundError as fnf_err:
                print(f"CRITICAL init_db ERROR: Schema file not found: {fnf_err}.")
            except MySQLError as err:
                print(f"CRITICAL init_db ERROR executing schema ({schema_path}): {err}"); temp_conn.rollback()
            except Exception as e:
                print(f"CRITICAL init_db UNEXPECTED ERROR initializing schema: {e}\n{traceback.format_exc()}"); temp_conn.rollback()
        else:
            print(f"init_db: Database table '{RESULTS_TABLE}' already exists.")

    except MySQLError as err:
        print(f"CRITICAL init_db ERROR during connection/setup: {err}")
        # Depending on the error, the application might not be able to function.
        # Log this severely or even exit if the DB is essential.
    except Exception as e:
        print(f"CRITICAL init_db UNEXPECTED error during DB init: {e}\n{traceback.format_exc()}")
    finally:
        if temp_cursor: temp_cursor.close()
        if temp_conn and temp_conn.is_connected(): temp_conn.close()
        print("--- init_db: Finished DB Initialization Check ---")


# --- Initialize DB on Application Start ---
# Run DB initialization logic when the app module is loaded by Gunicorn/Flask.
# This ensures the database and table are checked/created before requests are handled.
print(">>> Application starting: Performing database initialization check...")
# Create an app context temporarily to access app.config during init_db
with app.app_context():
    init_db()
print(">>> Application starting: Database initialization check complete.")


# --- Data Loading Function (Corrected Path Handling) ---
def load_stereotype_data(relative_filepath=CSV_FILE_PATH):
    """Loads stereotype data from the CSV file path relative to the app's root."""
    stereotype_data = []
    # app.root_path is the directory where this app.py file is located
    full_filepath = os.path.join(app.root_path, relative_filepath)
    print(f"--- load_stereotype_data: Attempting to load stereotype data from: {full_filepath} ---")
    try:
        if not os.path.exists(full_filepath):
            raise FileNotFoundError(f"Stereotype definition file not found at: {full_filepath}")

        with open(full_filepath, mode='r', encoding='utf-8-sig') as infile: # Use utf-8-sig for potential BOM
            reader = csv.DictReader(infile)
            required_cols = ['State', 'Category', 'Superset', 'Subsets'] # Define expected columns

            # Check if header exists and contains required columns
            if not reader.fieldnames or not all(field in reader.fieldnames for field in required_cols):
                 actual_cols = reader.fieldnames or []
                 missing = [c for c in required_cols if c not in actual_cols]
                 raise ValueError(f"CSV missing required columns: {missing}. Found columns: {actual_cols}")

            row_count = 0
            error_count = 0
            for i, row in enumerate(reader):
                row_count += 1
                # Check for None or empty strings, provide defaults or skip
                try:
                    state = row.get('State','').strip()
                    category = row.get('Category','').strip() or 'Uncategorized' # Default if empty
                    superset = row.get('Superset','').strip()
                    subsets_str = row.get('Subsets','') # Can be empty

                    if not state or not superset:
                         print(f"Warning: Skipping CSV row {i+2} due to missing State or Superset: {row}")
                         error_count += 1
                         continue # Skip row if essential data is missing

                    # Process subsets: split by comma, strip whitespace, remove empty strings, sort
                    subsets = sorted([s.strip() for s in subsets_str.split(',') if s.strip()])

                    stereotype_data.append({
                        'state': state,
                        'category': category,
                        'superset': superset,
                        'subsets': subsets
                    })
                except Exception as row_err:
                    # Log error for the specific row but continue processing others
                    print(f"Error processing CSV row {i+2}: {row_err}. Row data: {row}")
                    error_count += 1
                    continue # Continue to next row

        if not stereotype_data and row_count > 0:
             print(f"Warning: Processed {row_count} rows from {full_filepath}, but loaded 0 valid stereotype entries. Check file content and format.")
        elif error_count > 0:
             print(f"Successfully loaded {len(stereotype_data)} stereotype entries from {full_filepath} ({error_count} rows skipped due to errors).")
        else:
             print(f"Successfully loaded {len(stereotype_data)} stereotype entries from {full_filepath}")
        return stereotype_data

    except FileNotFoundError as fnf_err:
        print(f"FATAL ERROR: Could not find stereotype data file: {fnf_err}. Application might not function correctly.")
        return [] # Return empty list on critical file error
    except ValueError as ve:
        print(f"FATAL ERROR processing CSV header or structure: {ve}. Check CSV format.")
        return []
    except Exception as e:
        print(f"FATAL ERROR loading stereotype data: {e}\n{traceback.format_exc()}")
        return []
    finally:
         print("--- load_stereotype_data: Finished loading stereotype data ---")


# --- Load Data & States ---
print(">>> Loading stereotype definitions...")
ALL_STEREOTYPE_DATA = load_stereotype_data()
# Derive states ONLY from successfully loaded data
INDIAN_STATES = sorted(list(set(item['state'] for item in ALL_STEREOTYPE_DATA))) if ALL_STEREOTYPE_DATA else []

if not ALL_STEREOTYPE_DATA or not INDIAN_STATES:
    print("\nCRITICAL WARNING: Stereotype data loading failed or produced no states. Quiz functionality will be limited or broken.\n")
    # Provide a fallback or clear indicator if states couldn't be loaded
    INDIAN_STATES = ["Error: State data unavailable"]
else:
    print(f">>> States available for selection based on loaded data: {INDIAN_STATES}")


# --- Data Processing Logic (from process_results.py) ---
def calculate_mean_offensiveness(series):
    """Helper: Calculates mean of non-negative ratings (>= 0), returns NaN if none exist."""
    valid_ratings = series[series >= 0] # Filter out -1 or other placeholders
    return valid_ratings.mean() if not valid_ratings.empty else np.nan

def generate_aggregated_data():
    """
    Loads raw results from DB, loads definitions from CSV, expands annotations,
    aggregates results, and returns the final DataFrame. Returns None on critical error.
    """
    print("--- [Processing] Starting data aggregation ---")
    db_conn_proc = None
    aggregated_df = None # Initialize result DataFrame
    try:
        print("[Processing] Connecting to DB to fetch raw results...")
        # Use app.config for DB connection details
        db_conn_proc = mysql.connector.connect(
            host=app.config['MYSQL_HOST'], user=app.config['MYSQL_USER'],
            password=app.config['MYSQL_PASSWORD'], database=app.config['MYSQL_DB'],
            port=app.config['MYSQL_PORT'], connection_timeout=10
        )
        if not db_conn_proc.is_connected():
            raise MySQLError("Processing: Failed to connect to MySQL.")

        # Fetch data using pandas read_sql_query for simplicity
        print(f"[Processing] Fetching all data from '{RESULTS_TABLE}' table...")
        results_df = pd.read_sql_query(f"SELECT * FROM {RESULTS_TABLE}", db_conn_proc)
        print(f"[Processing] Loaded {len(results_df)} raw results.")

        if results_df.empty:
            print("[Processing] Raw results table is empty. Returning empty DataFrame.")
            return pd.DataFrame() # Return empty DataFrame, not None

        # Add Stereotype_State column based on user_state (as per original logic)
        results_df['Stereotype_State'] = results_df['user_state']

        # Load stereotype definitions again for processing (ensure path is correct)
        stereotypes_path = os.path.join(app.root_path, CSV_FILE_PATH)
        if not os.path.exists(stereotypes_path):
            raise FileNotFoundError(f"Processing: Stereotypes definition file not found at {stereotypes_path}")
        print(f"[Processing] Loading definitions for processing from: {stereotypes_path}")
        stereotypes_df = pd.read_csv(stereotypes_path, encoding='utf-8-sig')
        # Ensure required columns exist in the loaded definitions DF
        required_def_cols = ['State', 'Category', 'Superset', 'Subsets']
        if not all(col in stereotypes_df.columns for col in required_def_cols):
             missing_cols = [col for col in required_def_cols if col not in stereotypes_df.columns]
             raise ValueError(f"Processing: Definitions CSV is missing required columns: {missing_cols}.")
        # Create the Subsets_List for lookup
        stereotypes_df['Subsets_List'] = stereotypes_df['Subsets'].fillna('').astype(str).apply(
            lambda x: sorted([s.strip() for s in x.split(',') if s.strip()])
        )
        print(f"[Processing] Loaded {len(stereotypes_df)} definitions.")
        # Create a lookup dictionary: (State, Category, Superset) -> [subset1, subset2]
        subset_lookup = stereotypes_df.set_index(['State', 'Category', 'Superset'])['Subsets_List'].to_dict()

        print("[Processing] Expanding annotations based on definitions...")
        expanded_rows = []
        processing_errors = 0
        for index, result_row in results_df.iterrows():
            # Extract necessary fields from the raw result row, handling potential None values
            state = result_row.get('Stereotype_State')
            category = result_row.get('category')
            superset = result_row.get('attribute_superset')
            annotation = result_row.get('annotation')
            # Handle potential missing/null numeric value for rating safely
            rating_val = result_row.get('offensiveness_rating')
            rating = int(rating_val) if pd.notna(rating_val) else -1 # Use -1 if missing/null/NaN

            # Ensure essential fields are present and not empty strings before processing
            if not all([state, category, superset, annotation]):
                print(f"Warning [Processing]: Skipping result row {index} due to missing key data: State='{state}', Cat='{category}', Super='{superset}', Anno='{annotation}'")
                processing_errors += 1
                continue

            # Append the original superset annotation
            expanded_rows.append({
                'Stereotype_State': state, 'Category': category, 'Attribute': superset,
                'annotation': annotation, 'offensiveness_rating': rating
            })

            # Look up and append rows for each associated subset
            subsets_list = subset_lookup.get((state, category, superset), [])
            for subset in subsets_list:
                expanded_rows.append({
                    'Stereotype_State': state, 'Category': category, 'Attribute': subset,
                    'annotation': annotation, 'offensiveness_rating': rating # Inherit annotation/rating
                })

        if processing_errors > 0:
             print(f"[Processing] Note: Skipped {processing_errors} rows during expansion due to missing data.")

        if not expanded_rows:
            print("[Processing] No rows generated after expansion. Returning empty DataFrame.")
            return pd.DataFrame() # Return empty DataFrame if expansion yielded nothing

        expanded_annotations_df = pd.DataFrame(expanded_rows)
        print(f"[Processing] Created {len(expanded_annotations_df)} expanded rows (including supersets and subsets).")

        print("[Processing] Aggregating expanded results...")
        # Group by the state, category, and specific attribute (superset or subset)
        grouped = expanded_annotations_df.groupby(['Stereotype_State', 'Category', 'Attribute'])

        # Perform aggregation: count votes for each annotation type, calculate avg offensiveness
        aggregated_data = grouped.agg(
            Stereotype_Votes=('annotation', lambda x: (x == 'Stereotype').sum()),
            Not_Stereotype_Votes=('annotation', lambda x: (x == 'Not a Stereotype').sum()),
            Not_Sure_Votes=('annotation', lambda x: (x == 'Not sure').sum()),
            # Use the helper function for average offensiveness, which handles non-positive values
            Average_Offensiveness=('offensiveness_rating', calculate_mean_offensiveness)
        ).reset_index() # Reset index to make group keys into columns

        # Format the average offensiveness to 2 decimal places
        aggregated_data['Average_Offensiveness'] = aggregated_data['Average_Offensiveness'].round(2)

        print(f"[Processing] Aggregation complete. Final aggregated DataFrame has {len(aggregated_data)} rows.")
        print("--- [Processing] Finished data aggregation successfully ---")
        aggregated_df = aggregated_data # Assign the result

    except FileNotFoundError as e:
        print(f"ERROR [Processing]: Input file not found: {e}")
        flash(f"Error during processing: Required data file not found. {e}", "error")
        aggregated_df = None # Indicate critical error
    except (MySQLError, pd.errors.DatabaseError) as e:
        print(f"ERROR [Processing]: Database connection or query error: {e}")
        flash(f"Error during processing: Database error occurred. {e}", "error")
        aggregated_df = None
    except KeyError as e:
        print(f"ERROR [Processing]: Missing expected column in DataFrame: {e}. Check raw data and definition file consistency.")
        flash(f"Error during processing: Data structure mismatch (Missing Column: {e}).", "error")
        aggregated_df = None
    except ValueError as e:
        print(f"ERROR [Processing]: Data format or value error: {e}")
        flash(f"Error during processing: Data format issue encountered. {e}", "error")
        aggregated_df = None
    except Exception as e:
        print(f"UNEXPECTED ERROR [Processing]:\n{traceback.format_exc()}")
        flash(f"An unexpected error occurred during data processing: {e}", "error")
        aggregated_df = None
    finally:
        # Ensure the connection is closed even if errors occurred
        if db_conn_proc and db_conn_proc.is_connected():
            db_conn_proc.close()
            print("[Processing] Closed DB connection used for processing.")

    return aggregated_df # Return the DataFrame or None


# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handles the initial user info form page. Name and state are mandatory."""
    if request.method == 'POST':
        user_name = request.form.get('name', '').strip()
        user_state = request.form.get('user_state') # State selected by user
        user_age = request.form.get('age','').strip()
        user_sex = request.form.get('sex','')

        # Validation
        errors = False
        if not user_name:
            flash('Name is required.', 'error')
            errors = True
        # Check against the dynamically loaded states (handle error case gracefully)
        valid_states = [s for s in INDIAN_STATES if not s.startswith("Error:")]
        if not user_state or user_state not in valid_states:
            flash('Please select a valid state from the list.', 'error')
            errors = True
        # Optional: Basic validation for age if entered
        if user_age and not user_age.isdigit():
             flash('Age must be a whole number (e.g., 25).', 'error')
             errors = True
        elif user_age and int(user_age) < 0: # Basic sanity check
             flash('Age cannot be negative.', 'error')
             errors = True

        # Optional: Validate sex if a specific format is expected
        # if user_sex and user_sex not in ['Male', 'Female', 'Other', 'Prefer not to say']:
        #     flash('Please select a valid option for sex.', 'error'); errors = True

        if errors:
            # Pass current INDIAN_STATES list and submitted form data back to template
            print(f"Index validation failed. Rendering index again. Errors: {errors}")
            return render_template('index.html', states=INDIAN_STATES, form_data=request.form)

        # Prepare user info for redirect as query parameters
        user_info = {'name': user_name, 'state': user_state, 'age': user_age, 'sex': user_sex}
        print(f"Index POST successful. Redirecting to quiz with user info: {user_info}")
        # Redirect to the quiz page, passing user info
        return redirect(url_for('quiz', **user_info))

    # GET request: Render the initial form
    # Pass the list of states loaded from the CSV
    return render_template('index.html', states=INDIAN_STATES, form_data={})


@app.route('/quiz')
def quiz():
    """Displays the quiz questions FILTERED by the user's selected state."""
    # Retrieve user info passed as query parameters from the index redirect
    user_info = {
        'name': request.args.get('name'),
        'state': request.args.get('state'),
        'age': request.args.get('age'),
        'sex': request.args.get('sex')
    }

    # Validate essential info received from URL parameters
    if not user_info['name'] or not user_info['state']:
         print("Redirecting to index: User name or state missing in quiz URL parameters.")
         flash('User information is missing. Please start over from the home page.', 'error')
         return redirect(url_for('index'))

    # Check if stereotype data loaded correctly; redirect if not
    if not ALL_STEREOTYPE_DATA or (INDIAN_STATES and INDIAN_STATES[0].startswith("Error:")):
        print("Redirecting to index: Stereotype data not available.")
        flash('Error: Stereotype definitions could not be loaded. Please contact the administrator.', 'error')
        return redirect(url_for('index'))

    # Filter quiz items based on the user's selected state
    target_state = user_info['state']
    filtered_quiz_items = [item for item in ALL_STEREOTYPE_DATA if item['state'] == target_state]

    print(f"Quiz route: Displaying quiz for user '{user_info['name']}', focusing on state '{target_state}'. Found {len(filtered_quiz_items)} relevant items.")
    if not filtered_quiz_items:
        # Inform user if no specific stereotypes were found for their chosen state in the data
        flash(f"No specific stereotype items were found for {target_state} in our current list.", 'info')
        # Still render the page, maybe with a message, or redirect as appropriate

    # Render the quiz template, passing the filtered items and user info
    return render_template('quiz.html', quiz_items=filtered_quiz_items, user_info=user_info)


@app.route('/submit', methods=['POST'])
def submit():
    """Handles the submission of the quiz answers to MySQL."""
    print("--- submit route: Received POST request ---")
    cursor = get_db()
    if not cursor:
        # get_db() should have flashed an error if connection failed
        print("Submit Error: Database connection failed (get_db returned None). Redirecting.")
        # Ensure flash message is set if get_db didn't already
        if not any(msg[1] == 'error' for msg in get_flashed_messages(with_categories=True)):
             flash("Database connection failed. Cannot submit results.", "error")
        return redirect(url_for('index')) # Redirect if no DB cursor

    db_connection = getattr(g, 'db', None) # Get the connection from context global `g`

    try:
        # Retrieve user info submitted via hidden fields in the quiz form
        user_name = request.form.get('user_name')
        user_state = request.form.get('user_state')
        user_age_str = request.form.get('user_age')
        user_sex = request.form.get('user_sex') or None # Use None if empty string

        # Basic validation for essential user info from the form
        if not user_name or not user_state:
            print("Submit Error: User name or state missing in submission form data.")
            flash("User information was missing in the submission. Cannot save results.", 'error')
            return redirect(url_for('index'))

        # Process age: convert to integer if present, handle non-numeric input
        user_age = None
        if user_age_str:
            try:
                user_age_val = int(user_age_str)
                if user_age_val >= 0: # Allow 0, reject negative
                     user_age = user_age_val
                else:
                     print(f"Warning: Negative age value submitted: '{user_age_str}'. Storing as NULL.")
                     flash("Age cannot be negative; it was not saved.", "warning")
            except ValueError:
                print(f"Warning: Invalid age value submitted: '{user_age_str}'. Storing as NULL.")
                flash("Age was not a valid number; it was not saved.", "warning")
                # user_age remains None

        print(f"Submit route: Processing submission for User: {user_name}, State: {user_state}, Age: {user_age}, Sex: {user_sex}")

        results_to_insert = []
        processed_indices = set() # Keep track of processed items

        # Iterate through all form fields to find annotation data
        item_count = 0
        for key in request.form:
            if key.startswith('annotation_'):
                item_count += 1
                # Extract the unique identifier for the stereotype item
                try:
                    # Assumes identifiers are like 'annotation_0', 'annotation_1' etc.
                    identifier = key.split('_')[-1]
                    if not identifier.isdigit(): # Basic check if identifier is numeric
                         print(f"Warning: Non-numeric identifier found in key: {key}. Skipping.")
                         continue
                    # Check if this item has already been processed (e.g., if form has duplicate names)
                    if identifier in processed_indices: continue
                except IndexError:
                    print(f"Warning: Malformed annotation key found: {key}. Skipping.")
                    continue

                # Mark this identifier as processed
                processed_indices.add(identifier)

                # Retrieve related fields for this stereotype item using the identifier
                superset = request.form.get(f'superset_{identifier}')
                category = request.form.get(f'category_{identifier}')
                annotation = request.form.get(key) # The annotation value itself (e.g., 'Stereotype')

                # Skip if essential data for this item is missing or empty
                if not all([superset, category, annotation]):
                    print(f"Warning: Skipping item with identifier '{identifier}' due to missing data: Superset='{superset}', Cat='{category}', Anno='{annotation}'.")
                    continue

                # Determine offensiveness rating: required only if annotation is 'Stereotype'
                # Use -1 as default/placeholder if not applicable or invalid, assuming DB column expects INT NOT NULL
                offensiveness = -1
                if annotation == 'Stereotype':
                    rating_key = f'offensiveness_{identifier}'
                    rating_str = request.form.get(rating_key)
                    if rating_str is not None and rating_str != '': # Check if input exists and is not empty
                        try:
                            offensiveness_val = int(rating_str)
                            # Enforce the expected range (0-5)
                            if 0 <= offensiveness_val <= 5:
                                offensiveness = offensiveness_val
                            else:
                                print(f"Warning: Offensiveness rating '{rating_str}' for item '{identifier}' is outside the valid range (0-5). Storing default (-1).")
                                # Keep default -1
                        except (ValueError, TypeError):
                            print(f"Warning: Invalid offensiveness rating '{rating_str}' for item '{identifier}'. Storing default (-1).")
                            # Keep default -1
                    else:
                        # This case means 'Stereotype' was selected but no offensiveness rating was provided
                        # Could happen if JS fails or form structure is wrong. Log it.
                        print(f"Warning: 'Stereotype' selected for item '{identifier}' but no valid offensiveness rating provided/found (key: {rating_key}). Storing default (-1).")
                        # Keep default -1 (or flash error if rating is strictly mandatory for 'Stereotype')
                        # flash(f"Offensiveness rating missing for stereotype '{superset}'.", "warning") # Optional user feedback

                # Append the processed data for this item to the list for bulk insertion
                results_to_insert.append({
                    'user_name': user_name,
                    'user_state': user_state,
                    'user_age': user_age, # Will be None if age was invalid or not provided
                    'user_sex': user_sex, # Will be None if not provided
                    'category': category,
                    'attribute_superset': superset,
                    'annotation': annotation,
                    'offensiveness_rating': offensiveness # Will be -1 if not applicable or invalid
                })

        print(f"Submit route: Found {item_count} potential items in form. Processed {len(processed_indices)} unique identifiers. Prepared {len(results_to_insert)} rows for insertion.")

        # Insert collected results into the database if any were processed
        if results_to_insert:
            # Use parameterized query (%s format for mysql.connector) to prevent SQL injection
            sql = f"""
                INSERT INTO {RESULTS_TABLE}
                (user_name, user_state, user_age, user_sex, category, attribute_superset, annotation, offensiveness_rating)
                VALUES (%(user_name)s, %(user_state)s, %(user_age)s, %(user_sex)s, %(category)s, %(attribute_superset)s, %(annotation)s, %(offensiveness_rating)s)
            """
            try:
                # Use executemany for efficient bulk insertion
                print(f"Submit route: Executing INSERT statement for {len(results_to_insert)} rows...")
                cursor.executemany(sql, results_to_insert)
                if db_connection:
                     db_connection.commit() # Commit the transaction
                     print(f"Submit route: Successfully inserted {cursor.rowcount} results into the database. Committed transaction.")
                     flash(f"Successfully submitted {len(results_to_insert)} responses. Thank you!", 'success')
                else:
                     # This case should ideally not happen if cursor exists, but handle defensively
                     print("Submit Error: DB connection object not found after getting cursor. Cannot commit.")
                     flash("Internal error: Could not commit results.", "error")
                     return redirect(url_for('index'))

            except MySQLError as db_err:
                 # Log the error and rollback transaction
                 print(f"DATABASE INSERT ERROR: {db_err}")
                 print(f"Data attempted: {results_to_insert[0] if results_to_insert else 'N/A'}") # Log first row for context
                 try:
                     if db_connection: db_connection.rollback()
                     print("Submit route: Rolled back transaction due to DB error.")
                 except Exception as rb_err: print(f"Error during rollback: {rb_err}")
                 flash("A database error occurred while saving your responses. Please try again.", 'error')
                 # Decide where to redirect on error - maybe back to quiz or index
                 return redirect(url_for('index')) # Redirecting to index for simplicity
            except Exception as e:
                 # Catch unexpected errors during DB operation
                 print(f"UNEXPECTED DB INSERT ERROR: {e}\n{traceback.format_exc()}")
                 try:
                     if db_connection: db_connection.rollback()
                 except Exception as rb_err: print(f"Error during rollback: {rb_err}")
                 flash("An unexpected error occurred while saving your data.", 'error')
                 return redirect(url_for('index'))
        else:
             # This case means the form was submitted, but no valid annotation data was parsed
             print("Warning: Submission received, but no valid results were parsed from the form data.")
             flash("No valid responses were found in your submission. Nothing was saved.", 'warning')
             # Redirect to thank you page even if nothing saved, as the user did perform the action.

        # Redirect to the thank you page on successful submission or if no data was parsed
        print("Submit route: Redirecting to thank_you page.")
        return redirect(url_for('thank_you'))

    except Exception as e: # Broad catch for unexpected errors in the route logic itself
        print(f"SUBMIT ROUTE UNEXPECTED ERROR: {e}\n{traceback.format_exc()}")
        flash("An unexpected error occurred during the submission process.", 'error')
        # Attempt to rollback if a transaction might be open (though less likely here)
        try:
            db_conn = getattr(g, 'db', None)
            if db_conn and db_conn.is_connected(): db_conn.rollback()
        except Exception as rb_err: print(f"Error during final rollback attempt in submit route exception handler: {rb_err}")
        return redirect(url_for('index'))


@app.route('/thank_you')
def thank_you():
    """Displays the thank you page."""
    return render_template('thank_you.html')


# --- Admin Routes ---
# !! SECURITY WARNING: These routes have NO authentication by default. !!
# !! Implement proper authentication (e.g., Flask-Login, HTTP Basic Auth behind proxy) !!
# !! BEFORE deploying this application to a public environment.        !!

@app.route('/admin')
def admin_view():
    """Displays the collected results from the MySQL database. (NEEDS AUTH)"""
    print("--- admin_view: Request received ---")
    # !! ADD AUTHENTICATION CHECK HERE !!
    # Example (very basic, replace with real auth):
    # from flask import session
    # if not session.get('is_admin'):
    #     print("Admin view: Unauthorized access attempt.")
    #     return "Unauthorized", 403

    cursor = get_db()
    if not cursor:
        print("Admin view Error: Database connection failed.")
        flash("Database connection failed. Cannot load admin view.", "error")
        return redirect(url_for('index')) # Or a dedicated error page

    results_data = []
    try:
        # Fetch all results, ordered by timestamp descending
        # Ensure your 'results' table has a timestamp column (e.g., DEFAULT CURRENT_TIMESTAMP)
        # If not, order by another relevant column like an ID
        # Consider adding a LIMIT clause if the table might become very large, e.g., LIMIT 1000
        query = f"SELECT * FROM {RESULTS_TABLE} ORDER BY timestamp DESC" # Assuming 'timestamp' column exists
        print(f"Admin view: Executing query: {query}")
        cursor.execute(query)
        results_data = cursor.fetchall() # Fetch all rows as dictionaries
        print(f"Admin view: Fetched {len(results_data)} results.")
    except MySQLError as err:
        # Handle cases like the table not existing yet or column name errors
        print(f"Admin view DB Error: {err}")
        if "Unknown column 'timestamp'" in str(err):
             print("Admin view Warning: 'timestamp' column not found for ordering. Fetching without specific order.")
             try:
                  fallback_query = f"SELECT * FROM {RESULTS_TABLE}"
                  print(f"Admin view: Executing fallback query: {fallback_query}")
                  cursor.execute(fallback_query)
                  results_data = cursor.fetchall()
                  print(f"Admin view: Fetched {len(results_data)} results using fallback query.")
             except MySQLError as inner_err:
                  print(f"Admin view DB Error (fallback): {inner_err}")
                  flash(f'Error fetching results data: {inner_err}', 'error')
        else:
             flash(f'Error fetching results data: {err}', 'error')
    except Exception as e:
         print(f"Admin view Unexpected Error: {e}\n{traceback.format_exc()}")
         flash('Unexpected error loading admin data.', 'error')

    # Render the admin template, passing the fetched results
    return render_template('admin.html', results=results_data)

@app.route('/admin/download_processed')
def download_processed_data():
    """Triggers data processing and sends aggregated results as CSV. (NEEDS AUTH)"""
    print("--- download_processed_data: Request received ---")
    # !! ADD AUTHENTICATION CHECK HERE !!

    aggregated_df = generate_aggregated_data() # Call the main processing function

    # Check the result from the processing function
    if aggregated_df is None:
        # Error occurred during processing, generate_aggregated_data should have flashed message
        print("Download Processed Error: Processing failed (returned None). Redirecting to admin.")
        return redirect(url_for('admin_view'))
    if aggregated_df.empty:
        print("Download Processed Info: Processing resulted in an empty DataFrame. No file to download.")
        flash("No data available to process or the results table is empty.", "warning")
        return redirect(url_for('admin_view'))

    # Proceed to generate CSV if processing was successful and produced data
    try:
        print(f"Download Processed: Processing successful. Aggregated DataFrame has {len(aggregated_df)} rows. Generating CSV...")
        # Use BytesIO as an in-memory buffer for the CSV file
        buffer = io.BytesIO()
        # Write DataFrame to the buffer as CSV, ensuring UTF-8 encoding
        aggregated_df.to_csv(buffer, index=False, encoding='utf-8')
        # Reset buffer position to the beginning for reading
        buffer.seek(0)

        download_filename = 'final_aggregated_stereotypes.csv'
        print(f"Download Processed: Sending aggregated CSV file ('{download_filename}') for download...")
        # Use Flask's send_file to send the buffer content as a downloadable file
        return send_file(
            buffer,
            mimetype='text/csv',
            download_name=download_filename, # Filename for the user
            as_attachment=True # Ensure browser prompts for download
        )
    except Exception as e:
        # Catch errors during CSV generation or file sending
        print(f"Download Processed Error generating or sending CSV:\n{traceback.format_exc()}")
        flash(f"Error creating download file: {e}", "error")
        return redirect(url_for('admin_view'))

@app.route('/admin/download_raw')
def download_raw_data():
    """Fetches all raw results and sends them as CSV. (NEEDS AUTH)"""
    print("--- download_raw_data: Request received ---")
    # !! ADD AUTHENTICATION CHECK HERE !!

    db_conn_raw = None # Initialize connection variable specifically for this download
    try:
        print("[Raw Download] Establishing new DB connection...")
        # Use app.config for connection details
        db_conn_raw = mysql.connector.connect(
            host=app.config['MYSQL_HOST'], user=app.config['MYSQL_USER'],
            password=app.config['MYSQL_PASSWORD'], database=app.config['MYSQL_DB'],
            port=app.config['MYSQL_PORT'], connection_timeout=10
        )
        if not db_conn_raw.is_connected():
            raise MySQLError("Raw Download: Failed to establish connection for raw data download.")

        print(f"[Raw Download] Fetching all data from '{RESULTS_TABLE}'...")
        # Fetch the entire table into a pandas DataFrame
        # Add ordering if desired (e.g., by timestamp or ID)
        query = f"SELECT * FROM {RESULTS_TABLE} ORDER BY timestamp DESC" # Assuming timestamp exists
        # query = f"SELECT * FROM {RESULTS_TABLE} ORDER BY id DESC" # Alternative if using 'id'
        print(f"[Raw Download] Executing query: {query}")
        raw_results_df = pd.read_sql_query(query, db_conn_raw)
        print(f"[Raw Download] Fetched {len(raw_results_df)} raw rows.")

        if raw_results_df.empty:
            print("[Raw Download] Info: Raw results table is empty. No data to download.")
            flash("The raw results table is currently empty. No data to download.", "warning")
            return redirect(url_for('admin_view'))

        # Generate CSV in memory
        print("[Raw Download] Generating CSV in memory...")
        buffer = io.BytesIO()
        raw_results_df.to_csv(buffer, index=False, encoding='utf-8')
        buffer.seek(0)

        download_filename = 'raw_quiz_results.csv'
        print(f"[Raw Download] Sending raw CSV file ('{download_filename}')...")
        # Send the file for download
        return send_file(
            buffer,
            mimetype='text/csv',
            download_name=download_filename,
            as_attachment=True
        )

    except (MySQLError, pd.errors.DatabaseError) as e:
        # Handle database connection errors or errors during pandas read_sql_query
        print(f"ERROR [Raw Download]: Database or Pandas error occurred: {e}")
        flash(f"Error fetching or reading raw data from the database: {e}", "error")
        return redirect(url_for('admin_view'))
    except Exception as e:
        # Catch any other unexpected errors
        print(f"UNEXPECTED ERROR [Raw Download]:\n{traceback.format_exc()}")
        flash(f"An unexpected error occurred while preparing the raw data download: {e}", "error")
        return redirect(url_for('admin_view'))
    finally:
        # Ensure the separate database connection used for download is closed
        if db_conn_raw and db_conn_raw.is_connected():
            db_conn_raw.close()
            print("[Raw Download] Closed DB connection used for raw download.")


# --- Main Execution Block (REMOVED FOR DEPLOYMENT) ---
# The following block is used for running the app with Flask's built-in
# development server (`python app.py`). It MUST be removed or commented out
# when deploying with a production WSGI server like Gunicorn, which imports
# the 'app' object directly using the command in the Procfile (`gunicorn app:app`).
# Gunicorn handles starting the server and managing worker processes.

# if __name__ == '__main__':
#     # The init_db() call was moved outside this block to run when the module loads.
#     print("Starting Flask development server (DO NOT USE IN PRODUCTION)...")
#     # Debug mode should be OFF in production. Set host='0.0.0.0' to be reachable.
#     # Use a port like 5000 or 8080 for development.
#     app.run(host='0.0.0.0', port=5000, debug=True) # Set debug=True ONLY for local dev