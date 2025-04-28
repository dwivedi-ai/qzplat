# stereotype_quiz_app/app.py

import os
import csv
import io
import traceback
from urllib.parse import urlparse # Essential for parsing the URL

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

# --- Read Configuration from Environment Variables ---

# SECRET_KEY: Essential for session management, flashing messages
# Read from environment, provide a default ONLY for local convenience (unsafe for prod)
SECRET_KEY = os.environ.get('SECRET_KEY', 'local-dev-secret-key-replace-in-prod')
if SECRET_KEY == 'local-dev-secret-key-replace-in-prod':
    print("WARNING: Using default SECRET_KEY. Set a proper SECRET_KEY environment variable for production!")

# MYSQL_URL: The single connection string provided by the environment (e.g., Railway)
MYSQL_URL = os.environ.get('MYSQL_URL')

# --- Parse MySQL URL and Configure App ---
db_config = {} # Dictionary to hold parsed database connection details
is_db_configured = False # Flag to track if DB setup was successful

if MYSQL_URL:
    print(f"--- Found MYSQL_URL environment variable. Attempting to parse... ---")
    try:
        parsed_url = urlparse(MYSQL_URL)
        if parsed_url.scheme != 'mysql':
            raise ValueError(f"Invalid scheme '{parsed_url.scheme}' in MYSQL_URL. Expected 'mysql'.")

        db_config['MYSQL_HOST'] = parsed_url.hostname
        db_config['MYSQL_USER'] = parsed_url.username
        db_config['MYSQL_PASSWORD'] = parsed_url.password # Can be None if not provided in URL
        db_config['MYSQL_DB'] = parsed_url.path[1:] if parsed_url.path else None # Remove leading '/'
        db_config['MYSQL_PORT'] = parsed_url.port

        # Validate essential components were parsed
        if not all([db_config.get('MYSQL_HOST'), db_config.get('MYSQL_USER'), db_config.get('MYSQL_DB')]):
            print(f"CRITICAL ERROR: MYSQL_URL parsed incompletely. Missing Host, User, or DB.")
            print(f"Parsed: Host={db_config.get('MYSQL_HOST')}, User={db_config.get('MYSQL_USER')}, "
                  f"Password={'Set' if db_config.get('MYSQL_PASSWORD') else 'Not Set'}, "
                  f"DB={db_config.get('MYSQL_DB')}, Port={db_config.get('MYSQL_PORT')}")
            db_config = {} # Reset config if essential parts are missing
        else:
            print("--- Database Configuration Parsed Successfully ---")
            print(f"  Host: {db_config.get('MYSQL_HOST')}")
            print(f"  User: {db_config.get('MYSQL_USER')}")
            print(f"  Password: {'Set' if db_config.get('MYSQL_PASSWORD') else 'Not Set'}")
            print(f"  Database: {db_config.get('MYSQL_DB')}")
            print(f"  Port: {db_config.get('MYSQL_PORT', 'Default (3306 assumed if not specified)')}")
            print("-------------------------------------------------")
            is_db_configured = True # Mark DB as configured

    except Exception as e:
        print(f"CRITICAL ERROR: Failed to parse MYSQL_URL: {MYSQL_URL}")
        print(f"Error details: {e}")
        print(traceback.format_exc())
        db_config = {} # Ensure config is empty on error
else:
    # This is the error you were seeing
    print("CRITICAL ERROR: MYSQL_URL environment variable not found.")
    print("The application requires MYSQL_URL to be set in the environment.")

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY

# Populate app.config ONLY if parsing was successful and essential components were found
if is_db_configured:
    app.config.update(db_config) # Add MYSQL_HOST, MYSQL_USER etc. to app.config
    app.config['DB_CONFIGURED'] = True
    print("Flask app configured with database details.")
else:
    app.config['DB_CONFIGURED'] = False
    print("Flask app database NOT configured due to missing/invalid MYSQL_URL.")

# --- Database Functions ---
def get_db():
    """Opens a new MySQL connection and cursor for the current request context."""
    if 'db' not in g:
        if not app.config.get('DB_CONFIGURED', False):
            # This state should ideally prevent routes needing DB from working fully
            # Log it clearly when get_db is called without config
            print("ERROR get_db: Attempted to get DB connection, but DB is not configured.")
            flash('Database is not configured. Please contact the administrator.', 'error')
            g.db = None
            g.cursor = None
            return None # Return None to indicate failure

        # Proceed with connection attempt using details from app.config
        try:
            g.db = mysql.connector.connect(
                host=app.config['MYSQL_HOST'],
                user=app.config['MYSQL_USER'],
                password=app.config.get('MYSQL_PASSWORD'), # Use get() as password might be None
                database=app.config['MYSQL_DB'],
                port=app.config.get('MYSQL_PORT', 3306), # Default port if not specified
                connection_timeout=10
            )
            g.cursor = g.db.cursor(dictionary=True) # Use dictionary cursor
            # print("DEBUG get_db: Database connection successful.") # Less verbose debug
        except MySQLError as err:
            print(f"ERROR get_db: MySQL connection failed: {err}")
            # Log details used for the failed connection attempt
            print(f"DEBUG: Connection attempt failed with Host={app.config.get('MYSQL_HOST')}, "
                  f"User={app.config.get('MYSQL_USER')}, DB={app.config.get('MYSQL_DB')}, "
                  f"Port={app.config.get('MYSQL_PORT')}")
            flash('Database connection error. Please try again later.', 'error')
            g.db = None
            g.cursor = None
        except Exception as e:
            print(f"UNEXPECTED ERROR in get_db during connect: {e}")
            print(traceback.format_exc())
            flash('An unexpected error occurred connecting to the database.', 'error')
            g.db = None
            g.cursor = None

    return getattr(g, 'cursor', None) # Return cursor if connection successful, else None

@app.teardown_appcontext
def close_db(error):
    """Closes the database connection and cursor at the end of the request."""
    cursor = g.pop('cursor', None)
    db = g.pop('db', None)
    if cursor:
        try:
            cursor.close()
        except Exception as e:
            print(f"Error closing cursor: {e}")
    if db and db.is_connected():
        try:
            db.close()
            # print("DEBUG close_db: Database connection closed.") # Less verbose
        except Exception as e:
            print(f"Error closing DB connection: {e}")
    if error:
        # Log any errors that occurred during request handling
        print(f"App context teardown error detected: {error}")

def init_db():
    """Initializes the database: creates DB (if needed/perms allow) and table from schema."""
    if not app.config.get('DB_CONFIGURED'):
        print("CRITICAL init_db: Skipping DB initialization because DB is not configured.")
        return

    # Use connection details from app.config
    db_host = app.config['MYSQL_HOST']
    db_user = app.config['MYSQL_USER']
    db_password = app.config.get('MYSQL_PASSWORD')
    db_port = app.config.get('MYSQL_PORT', 3306)
    db_name = app.config['MYSQL_DB']

    print(f"--- init_db: Starting Initialization Check ---")
    print(f"  Target DB: {db_name} on {db_host}:{db_port}")

    temp_conn = None
    temp_cursor = None
    try:
        # 1. Connect to MySQL server (without selecting the database initially)
        print("init_db: Connecting to MySQL server...")
        temp_conn = mysql.connector.connect(
            host=db_host, user=db_user, password=db_password, port=db_port, connection_timeout=15
        )
        temp_cursor = temp_conn.cursor()
        print("init_db: Connected to server.")

        # 2. Create database if it doesn't exist (best effort, might fail on permissions)
        try:
            # Use backticks for safety, ensure UTF8MB4 for full unicode support
            temp_cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            temp_conn.commit()
            print(f"init_db: Database '{db_name}' checked/created (if permissions allowed).")
        except MySQLError as err:
            # Log as warning, maybe DB already exists or user lacks CREATE DATABASE privs
            print(f"Warning init_db: Could not execute CREATE DATABASE IF NOT EXISTS for '{db_name}'. "
                  f"This might be okay if DB exists or due to permissions. Error: {err}")

        # 3. Select the database
        try:
            temp_cursor.execute(f"USE `{db_name}`")
            print(f"init_db: Successfully selected database '{db_name}'.")
        except MySQLError as err:
            print(f"CRITICAL init_db ERROR: Failed to select database '{db_name}'. Error: {err}")
            print("Cannot proceed with table creation. Check DB name and user privileges.")
            return # Stop initialization if we can't USE the database

        # 4. Check if the results table exists
        print(f"init_db: Checking if table '{RESULTS_TABLE}' exists...")
        temp_cursor.execute(f"SHOW TABLES LIKE '{RESULTS_TABLE}'")
        table_exists = temp_cursor.fetchone()

        # 5. If table doesn't exist, execute the schema file
        if not table_exists:
            print(f"init_db: Table '{RESULTS_TABLE}' not found. Executing schema from '{SCHEMA_FILE}'...")
            schema_path = os.path.join(app.root_path, SCHEMA_FILE)
            if not os.path.exists(schema_path):
                 # Try alternative common location if the path is just the filename
                 schema_path_alt = os.path.join(app.root_path, 'schema_mysql.sql')
                 if os.path.exists(schema_path_alt):
                     schema_path = schema_path_alt
                 else:
                    print(f"CRITICAL init_db ERROR: Schema file '{SCHEMA_FILE}' not found at '{schema_path}' or '{schema_path_alt}'.")
                    raise FileNotFoundError(f"Schema file missing: {SCHEMA_FILE}")

            try:
                with open(schema_path, mode='r', encoding='utf-8') as f:
                    # Read the entire script
                    sql_script = f.read()

                # Execute the script (potentially multiple statements)
                # Use multi=True for mysql.connector
                print(f"init_db: Executing SQL script from {schema_path}...")
                statement_results = temp_cursor.execute(sql_script, multi=True)

                # Important: Iterate through results to consume them and handle potential errors
                stmt_count = 0
                for result in statement_results:
                    stmt_count += 1
                    # print(f"  Executed statement {stmt_count}. Rows affected: {result.rowcount}")
                    if result.with_rows:
                        # print(f"  Statement {stmt_count} produced rows. Fetching to clear.")
                        result.fetchall() # Must fetch results if any were generated

                temp_conn.commit() # Commit transaction after executing all statements
                print(f"init_db: Schema executed successfully. Table '{RESULTS_TABLE}' should now exist.")
            except FileNotFoundError as e:
                 # Already handled above, but catch again just in case
                 print(f"CRITICAL init_db ERROR: Schema file missing: {e}")
                 # No rollback needed as nothing was likely executed
            except MySQLError as err:
                print(f"CRITICAL init_db ERROR: Failed executing schema SQL: {err}")
                print(f"Attempting to rollback changes...")
                try: temp_conn.rollback()
                except Exception as rb_err: print(f"Rollback failed: {rb_err}")
            except Exception as e:
                print(f"CRITICAL init_db UNEXPECTED ERROR executing schema: {e}")
                print(traceback.format_exc())
                try: temp_conn.rollback()
                except Exception as rb_err: print(f"Rollback failed: {rb_err}")
        else:
            print(f"init_db: Table '{RESULTS_TABLE}' already exists. No schema execution needed.")

    except MySQLError as e:
        print(f"CRITICAL init_db ERROR during connection/setup phase: {e}")
    except Exception as e:
        print(f"CRITICAL init_db UNEXPECTED error during initialization: {e}")
        print(traceback.format_exc())
    finally:
        # Ensure cursor and connection are closed reliably
        if temp_cursor:
            try: temp_cursor.close()
            except Exception as e_close: print(f"Warning: Error closing init_db cursor: {e_close}")
        if temp_conn and temp_conn.is_connected():
            try: temp_conn.close()
            except Exception as e_close: print(f"Warning: Error closing init_db connection: {e_close}")
        print("--- init_db: Finished Initialization Check ---")


# --- Initialize DB on Application Start ---
# Perform initialization check only if the app was configured with DB details
if app.config.get('DB_CONFIGURED'):
    print(">>> Application starting: Performing database initialization check...")
    # Use app_context to ensure 'g' is available if needed by extensions (though init_db doesn't use g)
    with app.app_context():
        init_db()
    print(">>> Application starting: Database initialization check complete.")
else:
    print(">>> Application starting: Skipping DB initialization check (DB not configured).")


# --- Data Loading Function (Load Stereotypes from CSV) ---
def load_stereotype_data(relative_filepath=CSV_FILE_PATH):
    """Loads stereotype definitions from the CSV file."""
    stereotype_data = []
    full_filepath = os.path.join(app.root_path, relative_filepath)
    print(f"--- load_stereotype_data: Attempting to load from: {full_filepath} ---")
    try:
        if not os.path.exists(full_filepath):
            raise FileNotFoundError(f"CSV file not found at {full_filepath}")

        with open(full_filepath, mode='r', encoding='utf-8-sig') as infile: # utf-8-sig handles BOM
            reader = csv.DictReader(infile)
            required_cols = ['State', 'Category', 'Superset', 'Subsets']
            if not reader.fieldnames or not all(col in reader.fieldnames for col in required_cols):
                raise ValueError(f"CSV missing required columns ({required_cols}). Found: {reader.fieldnames}")

            processed_count = 0
            for i, row in enumerate(reader):
                try:
                    state = row.get('State','').strip()
                    category = row.get('Category','').strip() or 'Uncategorized' # Default category
                    superset = row.get('Superset','').strip()
                    subsets_str = row.get('Subsets','').strip() # Get subsets as string

                    # Basic validation for essential fields
                    if not state or not superset:
                        print(f"Warning load_stereotype_data: Skipping row {i+2} due to missing State or Superset.")
                        continue

                    # Process subsets: split by comma, strip whitespace, remove empty, sort
                    subsets = sorted([s.strip() for s in subsets_str.split(',') if s.strip()])

                    stereotype_data.append({
                        'state': state,
                        'category': category,
                        'superset': superset,
                        'subsets': subsets # Store subsets as a list
                    })
                    processed_count += 1
                except Exception as row_err:
                    print(f"Error processing row {i+2} in CSV: {row_err}. Row data: {row}")
                    continue # Skip row on error

        print(f"--- load_stereotype_data: Successfully loaded {processed_count} entries. ---")
        return stereotype_data

    except FileNotFoundError as e:
        print(f"FATAL load_stereotype_data: {e}")
        flash("Error: Stereotype definitions file not found. Quiz functionality may be limited.", "error")
        return []
    except ValueError as e:
        print(f"FATAL load_stereotype_data: CSV format error: {e}")
        flash("Error: Invalid format in stereotype definitions file.", "error")
        return []
    except Exception as e:
        print(f"FATAL load_stereotype_data: Unexpected error loading stereotypes: {e}")
        print(traceback.format_exc())
        flash("Error: Could not load stereotype definitions.", "error")
        return []

# --- Load Data & States on App Start ---
print(">>> Loading stereotype definitions...")
ALL_STEREOTYPE_DATA = load_stereotype_data() # Load data into memory
INDIAN_STATES = sorted(list(set(item['state'] for item in ALL_STEREOTYPE_DATA))) if ALL_STEREOTYPE_DATA else []
if not ALL_STEREOTYPE_DATA or not INDIAN_STATES:
    print("CRITICAL WARNING: Stereotype data loading failed or CSV is empty. Quiz may not function correctly.")
    # Provide a fallback or clear indicator if loading fails
    INDIAN_STATES = ["Error: State data unavailable"]
else:
    print(f">>> Stereotype definitions loaded for {len(INDIAN_STATES)} states.")


# --- Data Processing Logic (for Admin Downloads) ---
def calculate_mean_offensiveness(series):
    """Helper for aggregation: Calculates mean of ratings >= 0."""
    valid_ratings = series[series >= 0] # Filter out -1 (or other invalid markers)
    return valid_ratings.mean() if not valid_ratings.empty else np.nan # Use NaN for no valid ratings

def generate_aggregated_data():
    """Connects to DB, fetches results, processes with definitions, and returns aggregated DataFrame."""
    print("--- generate_aggregated_data: Starting ---")
    aggregated_df = pd.DataFrame() # Default to empty DataFrame

    if not app.config.get('DB_CONFIGURED'):
        print("ERROR generate_aggregated_data: Database not configured.")
        flash("Cannot generate data: Database is not configured.", "error")
        return None # Return None to indicate failure clearly

    # Use a dedicated connection for this potentially long-running task
    db_conn_proc = None
    try:
        print("generate_aggregated_data: Connecting to DB...")
        db_conn_proc = mysql.connector.connect(
            host=app.config['MYSQL_HOST'], user=app.config['MYSQL_USER'],
            password=app.config.get('MYSQL_PASSWORD'), database=app.config['MYSQL_DB'],
            port=app.config.get('MYSQL_PORT', 3306), connection_timeout=15
        )
        if not db_conn_proc.is_connected():
             raise MySQLError("Processing: Failed to establish database connection.")
        print("generate_aggregated_data: Connected. Fetching raw results...")

        # Fetch raw results
        results_df = pd.read_sql_query(f"SELECT * FROM {RESULTS_TABLE}", db_conn_proc)
        print(f"generate_aggregated_data: Fetched {len(results_df)} raw results.")

        if results_df.empty:
            print("generate_aggregated_data: No results found in the database.")
            return aggregated_df # Return empty DataFrame

        # Ensure required columns exist (adjust column names if they differ in your DB table)
        required_result_cols = ['user_state', 'category', 'attribute_superset', 'annotation', 'offensiveness_rating']
        if not all(col in results_df.columns for col in required_result_cols):
             missing_cols = [col for col in required_result_cols if col not in results_df.columns]
             raise ValueError(f"Database results table is missing required columns: {missing_cols}")

        # Use user_state from results as the key for matching stereotype definitions
        results_df['Stereotype_State'] = results_df['user_state']

        # Load stereotype definitions again for processing (or ensure ALL_STEREOTYPE_DATA is accessible)
        # If ALL_STEREOTYPE_DATA is reliable, use it directly
        if not ALL_STEREOTYPE_DATA:
             raise ValueError("Stereotype definitions were not loaded correctly earlier.")

        # Create a lookup structure from ALL_STEREOTYPE_DATA for efficient matching
        # Key: (State, Category, Superset) -> Value: List of Subsets
        subset_lookup = {}
        for item in ALL_STEREOTYPE_DATA:
            key = (item['state'], item['category'], item['superset'])
            subset_lookup[key] = item['subsets']

        print("generate_aggregated_data: Expanding results based on subsets...")
        expanded_rows = []
        processing_errors = 0
        for index, result_row in results_df.iterrows():
            try:
                state = result_row.get('Stereotype_State')
                category = result_row.get('category')
                superset = result_row.get('attribute_superset')
                annotation = result_row.get('annotation')
                # Ensure rating is integer, handle potential NaN/None before conversion
                rating_val = result_row.get('offensiveness_rating')
                rating = int(rating_val) if pd.notna(rating_val) else -1 # Default to -1 if missing/invalid

                # Basic check for necessary data in the row
                if not all([state, category, superset, annotation is not None]):
                    processing_errors += 1
                    continue

                # Add the superset attribute itself
                expanded_rows.append({
                    'Stereotype_State': state, 'Category': category, 'Attribute': superset,
                    'annotation': annotation, 'offensiveness_rating': rating
                })

                # Find corresponding subsets from the lookup
                subsets_list = subset_lookup.get((state, category, superset), [])
                # Add each subset attribute
                for subset in subsets_list:
                    expanded_rows.append({
                        'Stereotype_State': state, 'Category': category, 'Attribute': subset,
                        'annotation': annotation, 'offensiveness_rating': rating
                    })
            except Exception as row_proc_err:
                 print(f"Error processing result row {index}: {row_proc_err}. Data: {result_row.to_dict()}")
                 processing_errors += 1

        if processing_errors > 0:
            print(f"Warning generate_aggregated_data: Skipped {processing_errors} rows during expansion due to errors or missing data.")

        if not expanded_rows:
            print("generate_aggregated_data: No valid rows after expansion.")
            return aggregated_df # Return empty DataFrame

        expanded_annotations_df = pd.DataFrame(expanded_rows)
        print(f"generate_aggregated_data: Expanded to {len(expanded_annotations_df)} rows including subsets. Aggregating...")

        # Group by State, Category, and the combined Attribute (Superset/Subset)
        grouped = expanded_annotations_df.groupby(['Stereotype_State', 'Category', 'Attribute'])

        # Aggregate counts and average offensiveness
        aggregated_data = grouped.agg(
            Stereotype_Votes=('annotation', lambda x: (x == 'Stereotype').sum()),
            Not_Stereotype_Votes=('annotation', lambda x: (x == 'Not a Stereotype').sum()),
            Not_Sure_Votes=('annotation', lambda x: (x == 'Not sure').sum()),
            # Use the helper function for mean calculation
            Average_Offensiveness=('offensiveness_rating', calculate_mean_offensiveness)
        ).reset_index()

        # Round the average offensiveness
        aggregated_data['Average_Offensiveness'] = aggregated_data['Average_Offensiveness'].round(2)

        print(f"generate_aggregated_data: Aggregation complete. Result has {len(aggregated_data)} rows.")
        aggregated_df = aggregated_data

    except (MySQLError, pd.errors.DatabaseError) as e:
        print(f"ERROR generate_aggregated_data: Database error: {e}")
        flash(f"Error generating aggregated data: Database issue.", "error")
        aggregated_df = None # Indicate failure
    except FileNotFoundError as e: # Should be caught earlier, but handle defensively
        print(f"ERROR generate_aggregated_data: File not found: {e}")
        flash(f"Error generating aggregated data: Missing required file.", "error")
        aggregated_df = None
    except KeyError as e: # If expected columns are missing after joins/processing
        print(f"ERROR generate_aggregated_data: Missing expected data column: {e}")
        flash(f"Error generating aggregated data: Data mismatch.", "error")
        aggregated_df = None
    except ValueError as e: # Catch validation errors (e.g., missing columns)
        print(f"ERROR generate_aggregated_data: Value error during processing: {e}")
        flash(f"Error generating aggregated data: {e}", "error")
        aggregated_df = None
    except Exception as e:
        print(f"UNEXPECTED ERROR generate_aggregated_data: {e}")
        print(traceback.format_exc())
        flash(f"Error generating aggregated data: An unexpected error occurred.", "error")
        aggregated_df = None
    finally:
        if db_conn_proc and db_conn_proc.is_connected():
            try: db_conn_proc.close()
            except Exception as e_close: print(f"Warning: Error closing processing connection: {e_close}")
        print("--- generate_aggregated_data: Finished ---")

    return aggregated_df # Return DataFrame or None


# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    # Display database status clearly on the index page
    db_status_ok = app.config.get('DB_CONFIGURED', False)
    if not db_status_ok:
         flash("Warning: Database is not configured. Data submission will not work.", "warning")

    form_data = {} # To repopulate form on error
    if request.method == 'POST':
        form_data = request.form # Store submitted data
        user_name = request.form.get('name', '').strip()
        user_state = request.form.get('user_state')
        user_age = request.form.get('age','').strip()
        user_sex = request.form.get('sex','') # Allow empty

        errors = False
        if not user_name:
            flash('Name is required.', 'error')
            errors = True
        # Use the loaded states (handle error case)
        valid_states = [s for s in INDIAN_STATES if not s.startswith("Error:")]
        if not user_state or user_state not in valid_states:
            flash('Please select a valid state.', 'error')
            errors = True
        if user_age: # Age is optional, but validate if provided
            try:
                age_val = int(user_age)
                if age_val < 0 or age_val > 130: # Basic sanity check
                    raise ValueError("Age out of reasonable range.")
            except (ValueError, TypeError):
                flash('Age must be a valid number (e.g., 30).', 'error')
                errors = True
        # No validation needed for sex (optional)

        if errors:
            # Re-render index with errors and form data
            return render_template('index.html', states=INDIAN_STATES, form_data=form_data, db_status_ok=db_status_ok)

        # If no errors, prepare user info for the quiz
        user_info = {
            'name': user_name,
            'state': user_state,
            'age': user_age if user_age else None, # Store as None if empty
            'sex': user_sex if user_sex else None   # Store as None if empty
        }
        # Redirect to quiz page, passing user info as query parameters
        return redirect(url_for('quiz', **user_info))

    # GET request: Render the index page
    return render_template('index.html', states=INDIAN_STATES, form_data=form_data, db_status_ok=db_status_ok)

@app.route('/quiz')
def quiz():
    # Retrieve user info from query parameters
    user_info = {
        'name': request.args.get('name'),
        'state': request.args.get('state'),
        'age': request.args.get('age'),
        'sex': request.args.get('sex')
    }

    # Validate essential user info exists
    if not user_info['name'] or not user_info['state']:
        flash('User information is missing. Please start again from the home page.', 'error')
        return redirect(url_for('index'))

    # Check if stereotype data loaded correctly
    if not ALL_STEREOTYPE_DATA or (INDIAN_STATES and INDIAN_STATES[0].startswith("Error:")):
        flash('Error: Stereotype definitions could not be loaded. Cannot display quiz.', 'error')
        return redirect(url_for('index'))

    # Filter quiz items based on the user's selected state
    target_state = user_info['state']
    filtered_quiz_items = [item for item in ALL_STEREOTYPE_DATA if item['state'] == target_state]

    if not filtered_quiz_items:
        flash(f"No stereotype items found for the selected state: {target_state}. Perhaps try another state?", 'info')
        # Decide if redirecting or showing an empty quiz page is better
        # return redirect(url_for('index')) # Option 1: Redirect back
        return render_template('quiz.html', quiz_items=[], user_info=user_info) # Option 2: Show empty quiz

    return render_template('quiz.html', quiz_items=filtered_quiz_items, user_info=user_info)

@app.route('/submit', methods=['POST'])
def submit():
    # First, check if DB is configured. If not, submission is impossible.
    if not app.config.get('DB_CONFIGURED', False):
        print("ERROR submit: Attempted submission, but DB is not configured.")
        flash("Submission failed: Database is not configured.", 'error')
        return redirect(url_for('index')) # Redirect, maybe to index

    cursor = get_db()
    # get_db() handles flashing errors if connection fails
    if not cursor:
        print("ERROR submit: Failed to get database cursor.")
        # No need to flash again, get_db should have done it.
        return redirect(url_for('index')) # Redirect on DB connection failure

    db_connection = getattr(g, 'db', None)
    if not db_connection or not db_connection.is_connected():
        print("ERROR submit: Database connection object not found or closed unexpectedly.")
        flash("Internal server error: Lost database connection.", "error")
        return redirect(url_for('index'))

    try:
        # --- Extract User Info ---
        user_name = request.form.get('user_name')
        user_state = request.form.get('user_state')
        user_age_str = request.form.get('user_age')
        user_sex = request.form.get('user_sex') or None # Store None if empty/missing

        # Validate essential user info (should match hidden fields in quiz form)
        if not user_name or not user_state:
            flash("User information was missing from the submission. Please try again.", 'error')
            return redirect(url_for('index')) # Redirect if basic info is lost

        # Process optional age (convert to int, handle errors/empty)
        user_age = None
        if user_age_str and user_age_str.strip():
            try:
                user_age = int(user_age_str.strip())
                if user_age < 0 or user_age > 130: # Basic sanity check
                     flash(f"Age '{user_age_str}' seems unlikely, but was recorded.", "warning") # Log/warn but accept?
                    # Or reject:
                    # flash("Invalid age provided.", 'warning')
                    # user_age = None # Reset if invalid
            except (ValueError, TypeError):
                flash(f"Invalid age format '{user_age_str}'. Age not recorded.", "warning")
                # user_age remains None

        # --- Process Quiz Responses ---
        results_to_insert = []
        processed_indices = set() # To avoid double processing if form has duplicate names

        # Loop through form keys to find annotations (e.g., 'annotation_0', 'annotation_1')
        for key in request.form:
            if key.startswith('annotation_'):
                try:
                    parts = key.split('_')
                    if len(parts) == 2 and parts[1].isdigit():
                        identifier = parts[1] # The index '0', '1', etc.

                        # Avoid double processing (e.g., if name="annotation_0" appears twice)
                        if identifier in processed_indices: continue
                        processed_indices.add(identifier)

                        # Get related hidden fields using the identifier
                        # These fields MUST exist in your quiz.html form for each item
                        category = request.form.get(f'category_{identifier}')
                        superset = request.form.get(f'superset_{identifier}')
                        annotation = request.form.get(key) # The selected radio button value

                        # Validate that we got all necessary parts for this item
                        if not all([category, superset, annotation]):
                            print(f"Warning submit: Missing data for item index {identifier}. "
                                  f"Category={category}, Superset={superset}, Annotation={annotation}. Skipping.")
                            continue

                        # Determine offensiveness rating (only if annotation is 'Stereotype')
                        offensiveness = -1 # Default: Not applicable or not rated
                        if annotation == 'Stereotype':
                            rating_key = f'offensiveness_{identifier}'
                            rating_str = request.form.get(rating_key)
                            if rating_str is not None and rating_str.isdigit():
                                try:
                                    rating_val = int(rating_str)
                                    if 0 <= rating_val <= 5:
                                        offensiveness = rating_val
                                    else:
                                        print(f"Warning submit: Invalid offensiveness rating '{rating_str}' "
                                              f"for item {identifier}. Expected 0-5. Storing -1.")
                                except ValueError:
                                    print(f"Warning submit: Could not parse offensiveness rating '{rating_str}' "
                                          f"for item {identifier}. Storing -1.")
                            elif rating_str: # Present but not digits
                                print(f"Warning submit: Non-numeric offensiveness rating '{rating_str}' "
                                      f"for item {identifier}. Storing -1.")
                            # else: No rating submitted for a 'Stereotype' - store -1

                        # Add validated data to the list for batch insertion
                        results_to_insert.append({
                            'user_name': user_name,
                            'user_state': user_state,
                            'user_age': user_age, # Can be None
                            'user_sex': user_sex, # Can be None
                            'category': category,
                            'attribute_superset': superset, # Store the superset linked to the annotation
                            'annotation': annotation,
                            'offensiveness_rating': offensiveness # Stored as -1 if not applicable/rated
                        })
                    else:
                        print(f"Warning submit: Malformed annotation key encountered: {key}")
                except Exception as item_err:
                    print(f"Error processing form item for key '{key}': {item_err}")
                    # Decide whether to continue processing other items or fail the request

        # --- Insert Data into Database ---
        if results_to_insert:
            print(f"Submit: Preparing to insert {len(results_to_insert)} results...")
            # Ensure column names in SQL match the keys in the dictionaries
            sql = f"""
                INSERT INTO {RESULTS_TABLE}
                (user_name, user_state, user_age, user_sex, category, attribute_superset, annotation, offensiveness_rating)
                VALUES
                (%(user_name)s, %(user_state)s, %(user_age)s, %(user_sex)s, %(category)s, %(attribute_superset)s, %(annotation)s, %(offensiveness_rating)s)
            """
            try:
                # Use executemany for efficient batch insertion
                cursor.executemany(sql, results_to_insert)
                db_connection.commit() # Commit the transaction
                print(f"Submit: Successfully inserted {cursor.rowcount} rows.") # rowcount with executemany might be cumulative or per-exec
                flash(f"Thank you! Your {len(results_to_insert)} responses have been recorded.", 'success')
            except MySQLError as db_err:
                print(f"CRITICAL submit DB INSERT ERROR: {db_err}")
                # Log first few records attempted for debugging
                print(f"Data attempted (first 2): {results_to_insert[:2]}")
                try:
                    db_connection.rollback() # Rollback on error
                    print("Submit: Transaction rolled back.")
                except Exception as rb_err:
                    print(f"Error during rollback attempt: {rb_err}")
                flash("A database error occurred while saving your responses. Please try again.", 'error')
                # Redirect back to index or maybe quiz? Index is safer.
                return redirect(url_for('index'))
            except Exception as e:
                print(f"CRITICAL submit UNEXPECTED INSERT ERROR: {e}")
                print(traceback.format_exc())
                try: db_connection.rollback()
                except Exception as rb_err: print(f"Error during rollback attempt: {rb_err}")
                flash("An unexpected error occurred while saving your data.", 'error')
                return redirect(url_for('index'))
        else:
            # This happens if the form was submitted but no valid annotation_* fields were found/processed
            print("Submit: No valid responses found in the submission.")
            flash("No responses were detected in your submission. Did you select an option for each item?", 'warning')
            # Redirect back to quiz might be appropriate here, passing user info again
            # Need to ensure user_info is available here if redirecting to quiz
            # return redirect(url_for('quiz', name=user_name, state=user_state, age=user_age_str, sex=user_sex))
            # Or just go to thank you page / index
            return redirect(url_for('index'))

        # If insertion successful, redirect to thank you page
        return redirect(url_for('thank_you'))

    except Exception as e:
        # Catch-all for unexpected errors in the main try block of the route
        print(f"CRITICAL submit ROUTE UNEXPECTED ERROR: {e}")
        print(traceback.format_exc())
        flash("An unexpected error occurred processing your submission.", 'error')
        # Attempt rollback if connection exists and might be in a transaction state
        db_conn = getattr(g, 'db', None)
        if db_conn and db_conn.is_connected():
            try:
                # Check if connection is in a transaction (may not be reliable across all connectors/versions)
                # if db_conn.in_transaction: # This attribute might not exist
                db_conn.rollback()
                print("Submit outer catch: Rollback attempted.")
            except Exception as rb_err:
                print(f"Rollback attempt failed in outer catch block: {rb_err}")
        return redirect(url_for('index'))

@app.route('/thank_you')
def thank_you():
    return render_template('thank_you.html')

@app.route('/admin')
def admin_view():
    if not app.config.get('DB_CONFIGURED'):
        flash("Database is not configured. Admin view unavailable.", "error")
        return render_template('admin.html', results=[], db_status_ok=False) # Show page with error

    cursor = get_db()
    if not cursor:
        # Error already flashed by get_db
        return render_template('admin.html', results=[], db_status_ok=False) # Show page with error

    results_data = []
    try:
        query = f"SELECT * FROM {RESULTS_TABLE} ORDER BY timestamp DESC" # Order by submission time
        cursor.execute(query)
        results_data = cursor.fetchall() # Fetch all results
    except MySQLError as err:
        print(f"Admin View DB Error fetching results: {err}")
        flash(f'Error fetching results: {err}', 'error')
    except NameError: # Should not happen if RESULTS_TABLE is defined globally
        print("Admin View Error: RESULTS_TABLE variable not defined.")
        flash('Configuration error: Results table name missing.', 'error')
    except Exception as e:
        print(f"Admin View Unexpected Error: {e}\n{traceback.format_exc()}")
        flash('An unexpected error occurred fetching admin data.', 'error')

    # Pass DB status to template
    return render_template('admin.html', results=results_data, db_status_ok=True)

@app.route('/admin/download_processed')
def download_processed_data():
    # generate_aggregated_data handles DB config check and errors, returns None on failure
    aggregated_df = generate_aggregated_data()

    if aggregated_df is None:
        # Flash message should have been set by generate_aggregated_data
        if not get_flashed_messages(category_filter=["error"]): # Add default error if none exists
            flash("Failed to generate processed data.", "error")
        return redirect(url_for('admin_view'))

    if aggregated_df.empty:
        flash("No data available to process and download.", "warning")
        return redirect(url_for('admin_view'))

    try:
        # Use BytesIO for in-memory CSV creation
        buffer = io.BytesIO()
        aggregated_df.to_csv(buffer, index=False, encoding='utf-8')
        buffer.seek(0) # Reset buffer position to the beginning

        download_name = 'final_aggregated_stereotypes.csv'
        print(f"Download Processed: Sending '{download_name}' ({buffer.getbuffer().nbytes} bytes)")
        return send_file(
            buffer,
            mimetype='text/csv',
            download_name=download_name,
            as_attachment=True # Prompt user to download
        )
    except Exception as e:
        print(f"Download Processed Data Error creating file: {e}\n{traceback.format_exc()}")
        flash(f"Error creating processed data file: {e}", "error")
        return redirect(url_for('admin_view'))

@app.route('/admin/download_raw')
def download_raw_data():
    if not app.config.get('DB_CONFIGURED'):
        print("ERROR download_raw: Database not configured.")
        flash("Cannot download raw data: Database is not configured.", "error")
        return redirect(url_for('admin_view'))

    # Use a dedicated connection for potentially large downloads
    db_conn_raw = None
    try:
        print("Download Raw: Connecting to DB...")
        db_conn_raw = mysql.connector.connect(
            host=app.config['MYSQL_HOST'], user=app.config['MYSQL_USER'],
            password=app.config.get('MYSQL_PASSWORD'), database=app.config['MYSQL_DB'],
            port=app.config.get('MYSQL_PORT', 3306), connection_timeout=15
        )
        if not db_conn_raw.is_connected():
             raise MySQLError("Raw Download: Failed to establish database connection.")

        query = f"SELECT * FROM {RESULTS_TABLE} ORDER BY timestamp DESC"
        print(f"Download Raw: Executing query: {query}")

        # Use pandas to read directly - efficient for large datasets
        raw_results_df = pd.read_sql_query(query, db_conn_raw)
        print(f"Download Raw: Fetched {len(raw_results_df)} rows.")

        if raw_results_df.empty:
            flash("No raw results found in the database to download.", "warning")
            return redirect(url_for('admin_view'))

        # Create CSV in memory using BytesIO
        buffer = io.BytesIO()
        raw_results_df.to_csv(buffer, index=False, encoding='utf-8')
        buffer.seek(0) # Rewind buffer

        download_name = 'raw_quiz_results.csv'
        print(f"Download Raw: Sending '{download_name}' ({buffer.getbuffer().nbytes} bytes)")
        return send_file(
            buffer,
            mimetype='text/csv',
            download_name=download_name,
            as_attachment=True
        )

    except (MySQLError, pd.errors.DatabaseError) as e:
        print(f"ERROR Download Raw DB: {e}")
        flash(f"Database error fetching raw data: {e}", "error")
        return redirect(url_for('admin_view'))
    except Exception as e:
        print(f"UNEXPECTED ERROR Download Raw: {e}\n{traceback.format_exc()}")
        flash(f"An unexpected error occurred preparing the raw data download: {e}", "error")
        return redirect(url_for('admin_view'))
    finally:
        # Ensure the dedicated connection is closed
        if db_conn_raw and db_conn_raw.is_connected():
            try: db_conn_raw.close()
            except Exception as e_close: print(f"Warning: Error closing raw download connection: {e_close}")

# Remove the __main__ block for deployment (Gunicorn/Waitress will run the app)
# if __name__ == '__main__':
#     # Use this block ONLY for local development with 'python app.py'
#     # Make sure to install python-dotenv: pip install python-dotenv
#     try:
#         from dotenv import load_dotenv
#         load_dotenv() # Load variables from .env file
#         print("Loaded environment variables from .env file for local development.")
#     except ImportError:
#         print("Warning: python-dotenv not installed. .env file will not be loaded.")
#         print("         Consider running 'pip install python-dotenv'")
#
#     # Check if MYSQL_URL was loaded (or set manually) before running
#     if not os.environ.get('MYSQL_URL'):
#         print("\nCRITICAL LOCAL DEV ERROR: MYSQL_URL not found in environment or .env file.")
#         print("Please create a .env file with MYSQL_URL='mysql://user:pass@host:port/db' or set it manually.")
#     else:
#         # You might want to re-run init_db manually when starting locally sometimes
#         # with app.app_context():
#         #     init_db()
#         app.run(debug=True, port=5000) # Run Flask dev server