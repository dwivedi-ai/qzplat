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

# Load environment variables from .env file if it exists (primarily for local dev)
load_dotenv()

# --- Configuration from Environment Variables ---

# Path for CSV (relative to app.py) - Make sure this is correct in your project structure
CSV_FILE_PATH = os.path.join('data', 'stereotypes.csv')
# Path for MySQL Schema file (relative to app.py)
SCHEMA_FILE = 'schema.sql'

# Flask Secret Key (Load from env var, generate a strong random one for production)
SECRET_KEY = os.environ.get('SECRET_KEY', 'default-fallback-secret-key-CHANGE-ME')
if SECRET_KEY == 'default-fallback-secret-key-CHANGE-ME':
    print("WARNING: SECRET_KEY is not set in environment variables. Using default (INSECURE).")

# --- MySQL Configuration (Load from Environment Variables) ---
MYSQL_HOST = os.environ.get('MYSQL_HOST')
MYSQL_USER = os.environ.get('MYSQL_USER')
MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD')
MYSQL_DB = os.environ.get('MYSQL_DB')
MYSQL_PORT = int(os.environ.get('MYSQL_PORT', 3306)) # Default MySQL port
RESULTS_TABLE = 'results' # Name of the table in the database

# Check if essential MySQL config is missing
if not all([MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB]):
    print("CRITICAL ERROR: Database configuration (MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB) missing in environment variables.")
    # Depending on your deployment, you might want to exit here or let Flask fail later.
    # For now, we'll let it proceed so Flask can start, but DB operations will fail.

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
# Store DB config in app.config for easier access
app.config['MYSQL_HOST'] = MYSQL_HOST
app.config['MYSQL_USER'] = MYSQL_USER
app.config['MYSQL_PASSWORD'] = MYSQL_PASSWORD
app.config['MYSQL_DB'] = MYSQL_DB
app.config['MYSQL_PORT'] = MYSQL_PORT

# --- Basic Logging Setup ---
# Gunicorn/PythonAnywhere often handle logging, but this provides a basic setup.
logging.basicConfig(level=logging.INFO) # Log INFO level and above
app.logger.setLevel(logging.INFO)
if app.debug: # More verbose logging if Flask debug mode is somehow enabled (shouldn't be in prod)
     app.logger.setLevel(logging.DEBUG)

app.logger.info("Flask application starting...")
if not all([MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB]):
    app.logger.error("Database configuration is incomplete. Database operations will fail.")


# --- Database Functions (Using app.config) ---

def get_db():
    """Opens a new MySQL database connection and cursor if none exist for the current request context."""
    if 'db' not in g:
        db_host = current_app.config['MYSQL_HOST']
        db_user = current_app.config['MYSQL_USER']
        db_password = current_app.config['MYSQL_PASSWORD']
        db_name = current_app.config['MYSQL_DB']
        db_port = current_app.config['MYSQL_PORT']

        if not all([db_host, db_user, db_password, db_name]):
             app.logger.error("Attempted DB connection with incomplete configuration.")
             flash('Database configuration error. Please contact the administrator.', 'error')
             g.db = None
             g.cursor = None
             return None

        try:
            g.db = mysql.connector.connect(
                host=db_host,
                user=db_user,
                password=db_password,
                database=db_name,
                port=db_port,
                connection_timeout=10 # Add a timeout
            )
            g.cursor = g.db.cursor(dictionary=True)
            app.logger.debug("MySQL connection established for request.")
        except MySQLError as err:
            app.logger.error(f"Error connecting to MySQL: {err}")
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
    if db and db.is_connected():
        db.close()
        app.logger.debug("MySQL connection closed for request.")
    if error:
        app.logger.error(f"App context teardown error: {error}")

def init_db_table():
    """
    Checks if the results table exists and creates it using the schema file if it doesn't.
    Assumes the database itself already exists (standard practice in production).
    This is often run *once* during deployment setup, not on every app start.
    """
    db_conn_init = None
    cursor_init = None
    db_host = current_app.config['MYSQL_HOST']
    db_user = current_app.config['MYSQL_USER']
    db_password = current_app.config['MYSQL_PASSWORD']
    db_name = current_app.config['MYSQL_DB']
    db_port = current_app.config['MYSQL_PORT']

    if not all([db_host, db_user, db_password, db_name]):
        app.logger.error("Cannot initialize DB table: Database configuration missing.")
        return False

    try:
        app.logger.info(f"Attempting connection to DB '{db_name}' for table initialization check...")
        db_conn_init = mysql.connector.connect(
            host=db_host, user=db_user, password=db_password, database=db_name, port=db_port
        )
        cursor_init = db_conn_init.cursor()
        app.logger.info(f"Connected to DB '{db_name}'. Checking for table '{RESULTS_TABLE}'...")

        cursor_init.execute(f"SHOW TABLES LIKE '{RESULTS_TABLE}'")
        if not cursor_init.fetchone():
            app.logger.warning(f"Table '{RESULTS_TABLE}' not found. Attempting creation using '{SCHEMA_FILE}'...")
            schema_path = os.path.join(current_app.root_path, SCHEMA_FILE)
            if not os.path.exists(schema_path):
                 # Try alternate path if schema.sql wasn't specified (less common now)
                 schema_path = os.path.join(current_app.root_path, 'schema_mysql.sql')

            try:
                with open(schema_path, mode='r', encoding='utf-8') as f: sql_script = f.read()
                # Execute potentially multiple statements
                for result in cursor_init.execute(sql_script, multi=True):
                     if result.with_rows: # Consume results if needed (e.g., from SELECTs in script)
                         result.fetchall()
                db_conn_init.commit()
                app.logger.info(f"Database table '{RESULTS_TABLE}' created successfully.")
                return True # Table created
            except FileNotFoundError:
                app.logger.error(f"Schema file ({SCHEMA_FILE} or schema_mysql.sql) not found at {schema_path}.")
                return False
            except MySQLError as err:
                app.logger.error(f"Error executing schema file '{schema_path}': {err}")
                db_conn_init.rollback()
                return False
            except Exception as e:
                app.logger.error(f"Unexpected error initializing schema: {e}", exc_info=True)
                db_conn_init.rollback()
                return False
        else:
            app.logger.info(f"Database table '{RESULTS_TABLE}' already exists. No action needed.")
            return True # Table exists
    except MySQLError as err:
        app.logger.error(f"Error during DB table initialization check/creation: {err}")
        return False
    except Exception as e:
        app.logger.error(f"Unexpected error during DB table initialization: {e}", exc_info=True)
        return False
    finally:
        if cursor_init: cursor_init.close()
        if db_conn_init and db_conn_init.is_connected(): db_conn_init.close()


# --- Data Loading Function (Using app.root_path for robustness) ---
def load_stereotype_data(relative_filepath=CSV_FILE_PATH):
    """Loads stereotype data from the CSV within the 'data' directory."""
    stereotype_data = []
    # Use app.root_path which is the directory containing app.py
    full_filepath = os.path.join(current_app.root_path, relative_filepath)
    app.logger.info(f"Attempting to load stereotype data from: {full_filepath}")
    try:
        if not os.path.exists(full_filepath):
            raise FileNotFoundError(f"File not found: {full_filepath}")

        with open(full_filepath, mode='r', encoding='utf-8-sig') as infile: # Handle potential BOM
            reader = csv.DictReader(infile)
            required_cols = ['State', 'Category', 'Superset', 'Subsets']
            if not reader.fieldnames or not all(field in reader.fieldnames for field in required_cols):
                 missing = [c for c in required_cols if c not in (reader.fieldnames or [])]
                 raise ValueError(f"CSV missing required columns: {missing}. Found: {reader.fieldnames}")

            for i, row in enumerate(reader):
                try:
                    state = row.get('State','').strip()
                    category = row.get('Category','Uncategorized').strip() # Default category
                    superset = row.get('Superset','').strip()
                    subsets_str = row.get('Subsets','')
                    if not state or not superset: continue # Skip if essential info missing
                    subsets = sorted([s.strip() for s in subsets_str.split(',') if s.strip()]) # Process subsets
                    stereotype_data.append({'state': state, 'category': category, 'superset': superset, 'subsets': subsets})
                except Exception as row_err:
                    app.logger.warning(f"Error processing CSV row {i+1}: {row_err}")
                    continue # Skip problematic row

        app.logger.info(f"Successfully loaded {len(stereotype_data)} stereotype entries from {full_filepath}")
        return stereotype_data

    except FileNotFoundError:
        app.logger.critical(f"FATAL ERROR: Stereotype CSV file not found at {full_filepath}. Application may not function correctly.")
        return []
    except ValueError as ve:
        app.logger.critical(f"FATAL ERROR processing stereotype CSV: {ve}")
        return []
    except Exception as e:
        app.logger.critical(f"FATAL ERROR loading stereotype data: {e}", exc_info=True)
        return []

# --- Load Data & States (Load once at startup) ---
# Use with app.app_context() to ensure access to current_app for paths if needed earlier
with app.app_context():
    ALL_STEREOTYPE_DATA = load_stereotype_data()
    if ALL_STEREOTYPE_DATA:
        INDIAN_STATES = sorted(list(set(item['state'] for item in ALL_STEREOTYPE_DATA)))
        app.logger.info(f"States loaded: {len(INDIAN_STATES)}")
    else:
        app.logger.error("Stereotype data loading failed! Check previous logs. Using fallback empty states list.")
        INDIAN_STATES = ["Error: Data Load Failed"]

# --- Data Processing Logic ---
def calculate_mean_offensiveness(series):
    """Helper: Calculates mean of non-negative ratings, returns NaN if none exist."""
    valid_ratings = series[series >= 0]
    return valid_ratings.mean() if not valid_ratings.empty else np.nan

def generate_aggregated_data():
    """
    Loads raw results from DB, loads definitions, expands annotations, aggregates,
    and returns the final DataFrame. Returns None on critical error.
    """
    app.logger.info("--- [Processing] Starting data aggregation ---")
    db_conn_proc = None
    try:
        db_host = current_app.config['MYSQL_HOST']
        db_user = current_app.config['MYSQL_USER']
        db_password = current_app.config['MYSQL_PASSWORD']
        db_name = current_app.config['MYSQL_DB']
        db_port = current_app.config['MYSQL_PORT']

        if not all([db_host, db_user, db_password, db_name]):
            app.logger.error("[Processing] Cannot proceed: Database configuration missing.")
            flash("Processing Error: Server configuration issue.", "error")
            return None

        app.logger.info("[Processing] Connecting to DB to fetch raw results...")
        db_conn_proc = mysql.connector.connect(
            host=db_host, user=db_user, password=db_password, database=db_name, port=db_port
        )
        if not db_conn_proc.is_connected():
            raise MySQLError("Processing: Failed to connect to MySQL.")

        # Fetch data using Pandas (handles connection closing)
        results_df = pd.read_sql_query(f"SELECT * FROM {RESULTS_TABLE}", db_conn_proc)
        app.logger.info(f"[Processing] Loaded {len(results_df)} raw results.")
        if results_df.empty:
            app.logger.info("[Processing] Raw results table is empty. Returning empty DataFrame.")
            return pd.DataFrame() # Return empty DataFrame, not None

        # --- Ensure consistent column naming ---
        # Adjust based on your actual DB schema if needed
        results_df.rename(columns={'user_state': 'UserStateFromDB'}, inplace=True, errors='ignore') # Example rename if needed
        results_df['Stereotype_State'] = results_df['UserStateFromDB'] # Assuming user_state is the column name
        # Add error handling if column doesn't exist
        if 'UserStateFromDB' not in results_df.columns and 'user_state' not in results_df.columns:
             app.logger.error("[Processing] Missing 'user_state' column in fetched results.")
             flash("Processing Error: Data structure mismatch in results.", "error")
             return None

        # Load stereotype definitions
        stereotypes_path = os.path.join(current_app.root_path, CSV_FILE_PATH)
        if not os.path.exists(stereotypes_path):
             raise FileNotFoundError(f"Processing: Stereotypes file not found at {stereotypes_path}")
        app.logger.info(f"[Processing] Loading definitions from: {stereotypes_path}")
        stereotypes_df = pd.read_csv(stereotypes_path, encoding='utf-8-sig')
        # Ensure required columns exist in stereotypes_df
        req_stereo_cols = ['State', 'Category', 'Superset', 'Subsets']
        if not all(col in stereotypes_df.columns for col in req_stereo_cols):
            missing_cols = [col for col in req_stereo_cols if col not in stereotypes_df.columns]
            app.logger.error(f"[Processing] Stereotypes CSV missing required columns: {missing_cols}")
            flash("Processing Error: Stereotypes definition file structure mismatch.", "error")
            return None

        stereotypes_df['Subsets_List'] = stereotypes_df['Subsets'].fillna('').astype(str).apply(lambda x: [s.strip() for s in x.split(',') if s.strip()])
        app.logger.info(f"[Processing] Loaded {len(stereotypes_df)} definitions.")
        subset_lookup = stereotypes_df.set_index(['State', 'Category', 'Superset'])['Subsets_List'].to_dict()

        app.logger.info("[Processing] Expanding annotations...")
        expanded_rows = []
        # Ensure results_df columns match expected names
        required_result_cols = ['Stereotype_State', 'category', 'attribute_superset', 'annotation', 'offensiveness_rating']
        if not all(col in results_df.columns for col in required_result_cols):
             missing_cols = [col for col in required_result_cols if col not in results_df.columns]
             app.logger.error(f"[Processing] Raw results data missing expected columns: {missing_cols}")
             flash("Processing Error: Raw results data structure mismatch.", "error")
             return None

        for index, result_row in results_df.iterrows():
            state = result_row['Stereotype_State']; category = result_row['category']
            superset = result_row['attribute_superset']; annotation = result_row['annotation']
            rating = result_row['offensiveness_rating']
            # Basic validation
            if not all([isinstance(state, str), isinstance(category, str), isinstance(superset, str), isinstance(annotation, str)]):
                app.logger.warning(f"[Processing] Skipping row {index} due to invalid data types.")
                continue

            expanded_rows.append({'Stereotype_State': state, 'Category': category, 'Attribute': superset, 'annotation': annotation, 'offensiveness_rating': rating})
            subsets_list = subset_lookup.get((state, category, superset), [])
            for subset in subsets_list:
                expanded_rows.append({'Stereotype_State': state, 'Category': category, 'Attribute': subset, 'annotation': annotation, 'offensiveness_rating': rating})

        if not expanded_rows:
            app.logger.info("[Processing] No valid rows to expand or process.")
            return pd.DataFrame() # Return empty DataFrame
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
        app.logger.info(f"[Processing] Aggregation complete. Result has {len(aggregated_data)} rows.")
        app.logger.info("--- [Processing] Finished data aggregation successfully ---")
        return aggregated_data

    except FileNotFoundError as e:
        app.logger.error(f"[Processing] Error: {e}")
        flash(f"Error: Input file not found during processing. {e}", "error")
        return None
    except MySQLError as e:
        app.logger.error(f"[Processing] Database Error: {e}")
        flash(f"Error: Database error during processing. {e}", "error")
        return None
    except KeyError as e: # More specific Pandas error
        app.logger.error(f"[Processing] Error: Missing expected column in DataFrame: {e}.")
        flash(f"Error: Data structure mismatch (Column: {e}). Check logs.", "error")
        return None
    except pd.errors.DatabaseError as e: # Pandas-specific DB error
         app.logger.error(f"[Processing] Pandas Database Error: {e}")
         flash(f"Error: Database interaction error during processing. {e}", "error")
         return None
    except Exception as e:
        app.logger.error(f"[Processing] Unexpected Error: {e}", exc_info=True)
        flash(f"An unexpected error occurred during data processing: {e}", "error")
        return None
    finally:
        if db_conn_proc and db_conn_proc.is_connected():
            db_conn_proc.close()
            app.logger.info("[Processing] Closed DB connection.")


# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    """Handles the initial user info form page."""
    if request.method == 'POST':
        user_name = request.form.get('name', '').strip()
        user_state = request.form.get('user_state')
        errors = False
        if not user_name:
            flash('Name is required.', 'error'); errors = True
        if not user_state or user_state not in INDIAN_STATES:
            # Avoid flashing raw list of states if load failed
            valid_states_available = INDIAN_STATES != ["Error: Data Load Failed"]
            flash(f'Please select a valid state{" from the list" if valid_states_available else ""}.', 'error')
            errors = True

        if errors:
            return render_template('index.html', states=INDIAN_STATES, form_data=request.form)

        user_info = {
            'name': user_name,
            'state': user_state,
            'age': request.form.get('age',''),
            'sex': request.form.get('sex','')
        }
        app.logger.info(f"Index POST successful for user '{user_name}'. Redirecting to quiz.")
        return redirect(url_for('quiz', **user_info))

    # GET request
    if "Error: Data Load Failed" in INDIAN_STATES:
         flash("Error loading state data. Quiz may be unavailable.", "warning")
    return render_template('index.html', states=INDIAN_STATES, form_data={})


@app.route('/quiz')
def quiz():
    """Displays the quiz questions FILTERED by the user's state."""
    user_info = {
        'name': request.args.get('name'),
        'state': request.args.get('state'),
        'age': request.args.get('age'),
        'sex': request.args.get('sex')
    }

    if not user_info['name'] or not user_info['state']:
         app.logger.warning("Redirecting to index: User name or state missing in quiz URL query parameters.")
         flash('User information missing. Please start again.', 'error')
         return redirect(url_for('index'))

    if not ALL_STEREOTYPE_DATA or "Error: Data Load Failed" in INDIAN_STATES:
        app.logger.error(f"Quiz page requested for {user_info['name']} but stereotype data failed to load.")
        flash('Critical Error: Stereotype data could not be loaded. Cannot display quiz.', 'error')
        return redirect(url_for('index')) # Or render an error page

    filtered_quiz_items = [item for item in ALL_STEREOTYPE_DATA if item['state'] == user_info['state']]
    app.logger.info(f"Displaying quiz for user '{user_info['name']}', state '{user_info['state']}'. Found {len(filtered_quiz_items)} items.")

    if not filtered_quiz_items:
        flash(f"No specific stereotypes found for {user_info['state']} in our current list.", 'info')
        # Decide if you want to show an empty quiz or redirect
        # return render_template('no_quiz_items.html', user_info=user_info) # Example

    return render_template('quiz.html', quiz_items=filtered_quiz_items, user_info=user_info)


@app.route('/submit', methods=['POST'])
def submit():
    """Handles the submission of the quiz answers to MySQL."""
    cursor = get_db()
    if not cursor:
        app.logger.error("Submit failed: Could not get DB cursor.")
        # flash message already set by get_db
        return redirect(url_for('index'))
    db_connection = g.db # Get the connection from g

    try:
        user_name = request.form.get('user_name','').strip()
        user_state = request.form.get('user_state')
        user_age_str = request.form.get('user_age')
        user_sex = request.form.get('user_sex') or None # Handle empty string as None

        if not user_name or not user_state:
            app.logger.warning("Submit failed: User name or state missing in form.")
            flash("User information was missing from the submission. Please try again.", 'error')
            return redirect(url_for('index')) # Or maybe back to quiz with info?

        try:
            user_age = int(user_age_str) if user_age_str and user_age_str.isdigit() else None
        except ValueError:
            app.logger.warning(f"Invalid age value received: {user_age_str}. Storing as NULL.")
            user_age = None

        results_to_insert = []
        processed_indices = set() # To handle potential duplicate identifiers if form is structured unusually

        # Iterate through form keys to find annotations
        for key in request.form:
            if key.startswith('annotation_'):
                # Extract identifier (e.g., 'state_category_superset' or just an index)
                identifier = key.replace('annotation_', '')
                if identifier in processed_indices: continue
                processed_indices.add(identifier)

                # Retrieve related fields using the same identifier
                superset = request.form.get(f'superset_{identifier}')
                category = request.form.get(f'category_{identifier}')
                annotation = request.form.get(key)

                # Basic validation for this item's core data
                if not all([superset, category, annotation]):
                    app.logger.warning(f"Skipping item with identifier '{identifier}': missing superset, category, or annotation.")
                    continue

                offensiveness = -1 # Default: Not applicable / Not rated
                if annotation == 'Stereotype':
                    rating_str = request.form.get(f'offensiveness_{identifier}')
                    try:
                        # Only parse if a rating was actually submitted
                        if rating_str is not None and rating_str.isdigit():
                            rating_val = int(rating_str)
                            if 0 <= rating_val <= 5:
                                offensiveness = rating_val
                            else:
                                app.logger.warning(f"Received out-of-range offensiveness rating ({rating_val}) for item '{identifier}'. Storing as -1.")
                                # Keep offensiveness = -1
                        # else: offensiveness remains -1 if no radio button selected for this stereotype
                    except ValueError:
                         app.logger.warning(f"Received non-integer offensiveness rating ('{rating_str}') for item '{identifier}'. Storing as -1.")
                         # Keep offensiveness = -1

                results_to_insert.append({
                    'user_name': user_name,
                    'user_state': user_state,
                    'user_age': user_age,
                    'user_sex': user_sex,
                    'category': category,
                    'attribute_superset': superset,
                    'annotation': annotation,
                    'offensiveness_rating': offensiveness
                })

        if results_to_insert:
            sql = f"""
                INSERT INTO {RESULTS_TABLE}
                (user_name, user_state, user_age, user_sex, category, attribute_superset, annotation, offensiveness_rating)
                VALUES (%(user_name)s, %(user_state)s, %(user_age)s, %(user_sex)s, %(category)s, %(attribute_superset)s, %(annotation)s, %(offensiveness_rating)s)
            """
            try:
                cursor.executemany(sql, results_to_insert)
                db_connection.commit() # Commit the transaction
                app.logger.info(f"Inserted {cursor.rowcount} results into the database for user '{user_name}'.")
                flash(f"Successfully submitted {len(results_to_insert)} responses. Thank you!", 'success')
            except MySQLError as db_err:
                 app.logger.error(f"DB Insert Error for user '{user_name}': {db_err}")
                 db_connection.rollback() # Rollback on error
                 flash("An error occurred while saving your responses. Please try again.", 'error')
                 # Redirecting to index might lose context, consider redirecting back to quiz
                 # or showing a specific error page. For now, index:
                 return redirect(url_for('index'))
            except Exception as e: # Catch other potential errors during DB interaction
                 app.logger.error(f"Unexpected DB Insert Error for user '{user_name}': {e}", exc_info=True)
                 try:
                     db_connection.rollback()
                 except Exception as rb_err:
                     app.logger.error(f"Error during rollback attempt: {rb_err}")
                 flash("An unexpected error occurred while saving your data. Please contact support.", 'error')
                 return redirect(url_for('index'))
        else:
             app.logger.warning(f"No valid results parsed from submission for user '{user_name}'.")
             flash("No valid responses were processed from your submission. Please ensure you answered some questions.", 'warning')

        return redirect(url_for('thank_you'))

    except Exception as e:
        # Broad catch for unexpected errors in the route logic itself
        app.logger.error(f"Unexpected error in /submit route: {e}", exc_info=True)
        flash("An unexpected error occurred processing your submission. Please try again.", 'error')
        return redirect(url_for('index'))


@app.route('/thank_you')
def thank_you():
    """Displays the thank you page."""
    return render_template('thank_you.html')


# --- Admin Routes ---
# ### PRODUCTION SECURITY WARNING ###
# These routes are NOT protected by authentication as per the request.
# Anyone knowing the URL can view and download ALL collected data.
# It is STRONGLY recommended to add authentication (e.g., Flask-HTTPAuth, Flask-Login)
# before deploying this to a public environment.

@app.route('/admin')
def admin_view():
    """Displays the collected results from the MySQL database."""
    app.logger.warning("Accessed unsecured /admin route.") # Log access
    cursor = get_db()
    if not cursor:
        # flash message set in get_db
        return redirect(url_for('index'))

    results_data = []
    try:
        # Fetch results ordered by most recent first
        cursor.execute(f"SELECT * FROM {RESULTS_TABLE} ORDER BY timestamp DESC")
        results_data = cursor.fetchall() # fetchall() is fine for moderate data size
        app.logger.info(f"Admin view: Fetched {len(results_data)} results.")
    except MySQLError as err:
        app.logger.error(f"Error fetching admin data: {err}")
        flash('Error fetching results data.', 'error')
    except Exception as e:
         app.logger.error(f"Unexpected error fetching admin data: {e}", exc_info=True)
         flash('Unexpected error loading admin data.', 'error')

    return render_template('admin.html', results=results_data)

@app.route('/admin/download_processed')
def download_processed_data():
    """Triggers data processing and sends aggregated results as CSV."""
    app.logger.warning("Accessed unsecured /admin/download_processed route.") # Log access
    app.logger.info("Download request received for processed data.")
    aggregated_df = generate_aggregated_data() # This function now returns None on error or empty pd.DataFrame

    if aggregated_df is None:
        app.logger.error("Processing failed, cannot generate download. Redirecting to admin.")
        # Flash message should already be set by generate_aggregated_data
        return redirect(url_for('admin_view'))
    if aggregated_df.empty:
        app.logger.warning("Processing returned empty DataFrame. No data to download.")
        flash("No data available to process or download.", "warning")
        return redirect(url_for('admin_view'))

    try:
        app.logger.info("Processing successful. Generating CSV in memory for download...")
        # Use BytesIO for binary data (required by send_file)
        buffer = io.BytesIO()
        # Write UTF-8 encoded data to the buffer
        buffer.write(aggregated_df.to_csv(index=False, encoding='utf-8').encode('utf-8'))
        buffer.seek(0)
        app.logger.info("Sending aggregated CSV file for download...")
        return send_file(
            buffer,
            mimetype='text/csv',
            download_name='final_aggregated_stereotypes.csv',
            as_attachment=True
        )
    except Exception as e:
        app.logger.error(f"Error generating/sending processed CSV: {e}", exc_info=True)
        flash(f"Error creating download file: {e}", "error")
        return redirect(url_for('admin_view'))

@app.route('/admin/download_raw')
def download_raw_data():
    """Fetches all raw results and sends them as CSV."""
    app.logger.warning("Accessed unsecured /admin/download_raw route.") # Log access
    app.logger.info("Raw data download request received.")
    db_conn_raw = None
    try:
        db_host = current_app.config['MYSQL_HOST']
        db_user = current_app.config['MYSQL_USER']
        db_password = current_app.config['MYSQL_PASSWORD']
        db_name = current_app.config['MYSQL_DB']
        db_port = current_app.config['MYSQL_PORT']

        if not all([db_host, db_user, db_password, db_name]):
            app.logger.error("[Raw Download] Cannot proceed: Database configuration missing.")
            flash("Raw Download Error: Server configuration issue.", "error")
            return redirect(url_for('admin_view'))

        app.logger.info("[Raw Download] Connecting to DB...")
        # Use pandas directly for simplicity here
        db_conn_raw_str = f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

        app.logger.info("[Raw Download] Fetching data...")
        raw_results_df = pd.read_sql_query(f"SELECT * FROM {RESULTS_TABLE} ORDER BY timestamp DESC", db_conn_raw_str)
        app.logger.info(f"[Raw Download] Fetched {len(raw_results_df)} raw rows.")

        if raw_results_df.empty:
            flash("Raw results table is empty. No data to download.", "warning")
            return redirect(url_for('admin_view'))

        app.logger.info("[Raw Download] Generating CSV in memory...")
        buffer = io.BytesIO()
        buffer.write(raw_results_df.to_csv(index=False, encoding='utf-8').encode('utf-8'))
        buffer.seek(0)
        app.logger.info("[Raw Download] Sending raw CSV file...")
        return send_file(
            buffer,
            mimetype='text/csv',
            download_name='raw_quiz_results.csv',
            as_attachment=True
        )

    except (MySQLError, pd.errors.DatabaseError) as e:
        app.logger.error(f"[Raw Download] DB/Pandas Error: {e}")
        flash(f"Error fetching or reading raw data: {e}", "error")
        return redirect(url_for('admin_view'))
    except Exception as e:
        app.logger.error(f"[Raw Download] Unexpected Error: {e}", exc_info=True)
        flash(f"An unexpected error occurred preparing the raw data download: {e}", "error")
        return redirect(url_for('admin_view'))
    # No finally block needed as pd.read_sql_query handles connection closing


# --- Main Execution (for Local Development ONLY) ---
if __name__ == '__main__':
    # This block is NOT used by production WSGI servers like Gunicorn/uWSGI
    app.logger.info("Running in local development mode (Flask development server).")

    # Check/Create DB Table on local startup for convenience
    # In production, this should be handled as a separate deployment step.
    with app.app_context():
        app.logger.info("Running initial DB table check/creation (local dev)...")
        if not init_db_table():
            app.logger.warning("DB table initialization failed or DB config is missing. App might not work correctly.")
            # You might want to exit here in local dev if DB is crucial
            # sys.exit("DB Initialization Failed")

    # Use environment variables for host/port if available
    host = os.environ.get('FLASK_RUN_HOST', '0.0.0.0')
    try:
        port = int(os.environ.get('FLASK_RUN_PORT', '5000'))
    except ValueError:
        port = 5000
        app.logger.warning(f"Invalid FLASK_RUN_PORT value. Using default {port}.")

    # Set debug=True ONLY if FLASK_ENV is 'development'
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    if debug_mode:
        app.logger.warning("Flask development server running in DEBUG mode.")
        app.logger.warning("DO NOT use debug mode in production deployment!")

    app.run(host=host, port=port, debug=debug_mode)