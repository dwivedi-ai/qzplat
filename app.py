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

# Flask Secret Key (Read from environment variable)
# Provide a default for local development if .env is not used
SECRET_KEY = os.environ.get('SECRET_KEY', 'a_very_weak_default_secret_key_change_me')

# --- MySQL Configuration (Read from environment variables) ---
# Provide local defaults matching your original setup for convenience if .env isn't loaded/present
MYSQL_HOST = os.environ.get('MYSQL_HOST', 'localhost')
MYSQL_USER = os.environ.get('MYSQL_USER', 'stereotype_user')
MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD', 'RespAI@2025') # !! Change default if needed !!
MYSQL_DB = os.environ.get('MYSQL_DB', 'stereotype_quiz_db')
# Read port from env, default to 3306, ensure it's an integer
MYSQL_PORT = int(os.environ.get('MYSQL_PORT', 3306))
RESULTS_TABLE = 'results' # Name of the table in the database

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY # Use the variable read from env/default

# Store DB config in app.config for consistent access throughout the app
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
                # Consider adding connection timeout: connect_timeout=10
            )
            g.cursor = g.db.cursor(dictionary=True) # Dictionary cursor for dict-like rows
        except MySQLError as err:
            print(f"Error connecting to MySQL: {err}")
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
        except Exception as e:
             print(f"Error closing DB connection: {e}") # Log error
    if error:
        print(f"App context teardown error detected: {error}") # Log error


def init_db():
    """Connects to MySQL, creates the database if needed, and creates the table using schema_mysql.sql."""
    temp_conn = None
    temp_cursor = None
    # Use app.config consistently
    db_host = app.config['MYSQL_HOST']
    db_user = app.config['MYSQL_USER']
    db_password = app.config['MYSQL_PASSWORD']
    db_port = app.config['MYSQL_PORT']
    db_name = app.config['MYSQL_DB']

    try:
        # First, connect without specifying a database to create it if needed
        print(f"Attempting initial connection to MySQL server at {db_host}:{db_port}...")
        temp_conn = mysql.connector.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            port=db_port
        )
        temp_cursor = temp_conn.cursor()
        print(f"Connected to MySQL server. Checking/creating database '{db_name}'...")
        try:
            # Use backticks for safety with potentially reserved keywords or special chars in DB name
            temp_cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            temp_conn.commit()
            print(f"Database '{db_name}' checked/created.")
            temp_cursor.execute(f"USE `{db_name}`") # Switch to the database
            print(f"Switched to database '{db_name}'.")
        except MySQLError as err:
            print(f"Failed to create/use database '{db_name}': {err}")
            # Consider if the app should stop here or try to continue if DB exists but USE failed
            return # Exit init_db if database cannot be used

        # Now check if the table exists within the selected database
        print(f"Checking for table '{RESULTS_TABLE}'...")
        temp_cursor.execute(f"SHOW TABLES LIKE '{RESULTS_TABLE}'")
        if not temp_cursor.fetchone():
            print(f"Table '{RESULTS_TABLE}' not found. Attempting creation using {SCHEMA_FILE}...")
            schema_path = os.path.join(app.root_path, SCHEMA_FILE)
            if not os.path.exists(schema_path):
                # Attempt alternate path if specific SCHEMA_FILE not found
                schema_path_alt = os.path.join(app.root_path, 'schema_mysql.sql')
                print(f"Warning: {SCHEMA_FILE} not found at {schema_path}. Trying {schema_path_alt}")
                schema_path = schema_path_alt # Use the alternative path

            try:
                if not os.path.exists(schema_path):
                     raise FileNotFoundError(f"Schema file not found at either {SCHEMA_FILE} or schema_mysql.sql")

                with open(schema_path, mode='r', encoding='utf-8') as f:
                    sql_script = f.read()
                # Execute the script, handling multiple statements separated by ';'
                print(f"Executing SQL script from {schema_path}...")
                # Use multi=True for scripts potentially containing multiple statements
                for result in temp_cursor.execute(sql_script, multi=True):
                    if result.with_rows:
                         print(f"Schema query result (rows): {result.fetchall()}")
                    else:
                         print(f"Schema query result (affected rows): {result.rowcount}")
                temp_conn.commit()
                print(f"Database table '{RESULTS_TABLE}' created successfully.")
            except FileNotFoundError as fnf_err:
                print(f"CRITICAL ERROR: Schema file not found: {fnf_err}.")
            except MySQLError as err:
                print(f"CRITICAL ERROR executing schema ({schema_path}): {err}"); temp_conn.rollback()
            except Exception as e:
                import traceback
                print(f"CRITICAL ERROR initializing schema: {e}\n{traceback.format_exc()}"); temp_conn.rollback()
        else:
            print(f"Database table '{RESULTS_TABLE}' already exists.")
    except MySQLError as err:
        print(f"CRITICAL ERROR during DB initialization connection/setup: {err}")
        # Depending on the error, the application might not be able to function.
        # Consider logging this severely or even exiting if the DB is essential.
    except Exception as e:
        import traceback
        print(f"CRITICAL UNEXPECTED error during DB init: {e}\n{traceback.format_exc()}")
    finally:
        if temp_cursor: temp_cursor.close()
        if temp_conn and temp_conn.is_connected(): temp_conn.close()
        print("DB initialization check finished.")

# --- Initialize DB (Moved Here) ---
# Run DB initialization logic when the app module is loaded by Gunicorn.
# This ensures the database and table are checked/created before requests are handled.
print("Application starting: Performing database initialization check...")
with app.app_context(): # Create an app context to access app.config during init
    init_db()
print("Application starting: Database initialization check complete.")

# --- Data Loading Function (Corrected Path Handling) ---
def load_stereotype_data(relative_filepath=CSV_FILE_PATH):
    """Loads stereotype data from the CSV file path relative to the app's root."""
    stereotype_data = []
    # app.root_path is the directory where app.py is located
    full_filepath = os.path.join(app.root_path, relative_filepath)
    print(f"Attempting to load stereotype data from: {full_filepath}")
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

            for i, row in enumerate(reader):
                # Check for None or empty strings, provide defaults or skip
                try:
                    state = row.get('State','').strip()
                    category = row.get('Category','').strip() or 'Uncategorized' # Default if empty
                    superset = row.get('Superset','').strip()
                    subsets_str = row.get('Subsets','') # Can be empty

                    if not state or not superset:
                         print(f"Warning: Skipping CSV row {i+2} due to missing State or Superset: {row}")
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
                    continue # Continue to next row

        if not stereotype_data:
             print(f"Warning: No valid stereotype entries loaded from {full_filepath}. Check file content and format.")
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
        import traceback
        print(f"FATAL ERROR loading stereotype data: {e}\n{traceback.format_exc()}")
        return []

# --- Load Data & States ---
print("Loading stereotype definitions...")
ALL_STEREOTYPE_DATA = load_stereotype_data()
# Derive states ONLY from successfully loaded data
INDIAN_STATES = sorted(list(set(item['state'] for item in ALL_STEREOTYPE_DATA))) if ALL_STEREOTYPE_DATA else []

if not ALL_STEREOTYPE_DATA or not INDIAN_STATES:
    print("\nCRITICAL WARNING: Stereotype data loading failed or produced no states. Quiz functionality will be limited or broken.\n")
    # Provide a fallback or clear indicator if states couldn't be loaded
    INDIAN_STATES = ["Error: State data unavailable"]
else:
    print(f"States available for selection based on loaded data: {INDIAN_STATES}")


# --- Data Processing Logic (from process_results.py) ---
# (Keep this section as it was in your original code, ensuring it uses app.config for DB details)
def calculate_mean_offensiveness(series):
    """Helper: Calculates mean of non-negative ratings, returns NaN if none exist."""
    valid_ratings = series[series >= 0] # Filter out -1 or other placeholders
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
        # Use app.config for DB connection details
        db_conn_proc = mysql.connector.connect(
            host=app.config['MYSQL_HOST'], user=app.config['MYSQL_USER'],
            password=app.config['MYSQL_PASSWORD'], database=app.config['MYSQL_DB'],
            port=app.config['MYSQL_PORT']
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
        if not {'State', 'Category', 'Superset', 'Subsets'}.issubset(stereotypes_df.columns):
             raise ValueError("Processing: Definitions CSV is missing required columns (State, Category, Superset, Subsets).")
        # Create the Subsets_List for lookup
        stereotypes_df['Subsets_List'] = stereotypes_df['Subsets'].fillna('').astype(str).apply(
            lambda x: sorted([s.strip() for s in x.split(',') if s.strip()])
        )
        print(f"[Processing] Loaded {len(stereotypes_df)} definitions.")
        # Create a lookup dictionary: (State, Category, Superset) -> [subset1, subset2]
        subset_lookup = stereotypes_df.set_index(['State', 'Category', 'Superset'])['Subsets_List'].to_dict()

        print("[Processing] Expanding annotations based on definitions...")
        expanded_rows = []
        for index, result_row in results_df.iterrows():
            # Extract necessary fields from the raw result row
            state = result_row.get('Stereotype_State') # Use the added column
            category = result_row.get('category')
            superset = result_row.get('attribute_superset')
            annotation = result_row.get('annotation')
            # Handle potential missing numeric value for rating
            rating_val = result_row.get('offensiveness_rating')
            rating = int(rating_val) if pd.notna(rating_val) else -1 # Use -1 if missing/null

            # Ensure essential fields are present before processing
            if not all([state, category, superset, annotation]):
                print(f"Warning [Processing]: Skipping result row {index} due to missing key data: {result_row.to_dict()}")
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

        # Format the average offensiveness
        aggregated_data['Average_Offensiveness'] = aggregated_data['Average_Offensiveness'].round(2)

        print(f"[Processing] Aggregation complete. Final aggregated DataFrame has {len(aggregated_data)} rows.")
        print("--- [Processing] Finished data aggregation successfully ---")
        return aggregated_data

    except FileNotFoundError as e:
        print(f"ERROR [Processing]: Input file not found: {e}")
        flash(f"Error during processing: Required data file not found. {e}", "error")
        return None # Indicate critical error
    except (MySQLError, pd.errors.DatabaseError) as e:
        print(f"ERROR [Processing]: Database connection or query error: {e}")
        flash(f"Error during processing: Database error occurred. {e}", "error")
        return None
    except KeyError as e:
        print(f"ERROR [Processing]: Missing expected column in DataFrame: {e}. Check raw data and definition file consistency.")
        flash(f"Error during processing: Data structure mismatch (Missing Column: {e}).", "error")
        return None
    except ValueError as e:
        print(f"ERROR [Processing]: Data format or value error: {e}")
        flash(f"Error during processing: Data format issue encountered. {e}", "error")
        return None
    except Exception as e:
        import traceback
        print(f"UNEXPECTED ERROR [Processing]:\n{traceback.format_exc()}")
        flash(f"An unexpected error occurred during data processing: {e}", "error")
        return None
    finally:
        # Ensure the connection is closed even if errors occurred
        if db_conn_proc and db_conn_proc.is_connected():
            db_conn_proc.close()
            print("[Processing] Closed DB connection used for processing.")


# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    """Handles the initial user info form page. Name is mandatory."""
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
        # Check against the dynamically loaded states (handle error case)
        valid_states = [s for s in INDIAN_STATES if not s.startswith("Error:")]
        if not user_state or user_state not in valid_states:
            flash('Please select a valid state from the list.', 'error')
            errors = True
        # Optional: Validate age if entered (e.g., must be numeric)
        if user_age and not user_age.isdigit():
             flash('Age must be a number.', 'error')
             errors = True
        # Optional: Validate sex if needed (e.g., must be Male/Female if required)
        # if user_sex and user_sex not in ['Male', 'Female']:
        #     flash('Please select a valid sex.', 'error'); errors = True

        if errors:
            # Pass current INDIAN_STATES list and submitted form data back to template
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
    if not ALL_STEREOTYPE_DATA or INDIAN_STATES[0].startswith("Error:"):
        print("Redirecting to index: Stereotype data not available.")
        flash('Error: Stereotype definitions could not be loaded. Please contact the administrator.', 'error')
        return redirect(url_for('index'))

    # Filter quiz items based on the user's selected state
    target_state = user_info['state']
    filtered_quiz_items = [item for item in ALL_STEREOTYPE_DATA if item['state'] == target_state]

    print(f"Displaying quiz for user '{user_info['name']}', focusing on state '{target_state}'. Found {len(filtered_quiz_items)} relevant stereotype items.")
    if not filtered_quiz_items:
        # Inform user if no specific stereotypes were found for their chosen state in the data
        flash(f"No specific stereotype items were found for {target_state} in our current list.", 'info')
        # Still render the page, maybe with a message, or redirect as appropriate

    # Render the quiz template, passing the filtered items and user info
    return render_template('quiz.html', quiz_items=filtered_quiz_items, user_info=user_info)


@app.route('/submit', methods=['POST'])
def submit():
    """Handles the submission of the quiz answers to MySQL."""
    cursor = get_db()
    if not cursor:
        # get_db() should have flashed an error if connection failed
        flash("Database connection failed. Cannot submit results.", "error")
        return redirect(url_for('index')) # Redirect if no DB cursor

    db_connection = g.db # Get the connection from context global `g`

    try:
        # Retrieve user info submitted via hidden fields in the quiz form
        user_name = request.form.get('user_name')
        user_state = request.form.get('user_state')
        user_age_str = request.form.get('user_age')
        user_sex = request.form.get('user_sex') or None # Use None if empty string

        # Basic validation for essential user info from the form
        if not user_name or not user_state:
            flash("User information was missing in the submission. Cannot save results.", 'error')
            return redirect(url_for('index')) # Or back to quiz if possible

        # Process age: convert to integer if present, handle non-numeric input
        user_age = None
        if user_age_str:
            try:
                user_age = int(user_age_str)
                if user_age < 0: user_age = None # Basic sanity check
            except ValueError:
                print(f"Warning: Invalid age value submitted: '{user_age_str}'. Storing as NULL.")
                flash("Age was not a valid number; it was not saved.", "warning")
                user_age = None # Store as NULL in DB if invalid

        results_to_insert = []
        processed_indices = set() # Keep track of processed items to avoid duplicates if form is structured weirdly

        # Iterate through all form fields to find annotation data
        for key in request.form:
            if key.startswith('annotation_'):
                # Extract the unique identifier for the stereotype item
                # Assumes identifiers are integers like 'annotation_0', 'annotation_1' etc.
                try:
                    identifier = key.split('_')[-1]
                    # Check if this item has already been processed (belt-and-suspenders)
                    if identifier in processed_indices: continue
                except IndexError:
                    print(f"Warning: Malformed annotation key found: {key}. Skipping.")
                    continue

                # Mark this identifier as processed
                processed_indices.add(identifier)

                # Retrieve related fields for this stereotype item using the identifier
                superset = request.form.get(f'superset_{identifier}')
                category = request.form.get(f'category_{identifier}')
                annotation = request.form.get(key) # The annotation value itself

                # Skip if essential data for this item is missing
                if not all([superset, category, annotation]):
                    print(f"Warning: Skipping item with identifier '{identifier}' due to missing data (superset, category, or annotation).")
                    continue

                # Determine offensiveness rating: required only if annotation is 'Stereotype'
                offensiveness = -1 # Default value for DB (use NULL if column allows, otherwise -1 or similar)
                if annotation == 'Stereotype':
                    rating_key = f'offensiveness_{identifier}'
                    rating_str = request.form.get(rating_key)
                    if rating_str is not None and rating_str != '': # Check if the input exists and is not empty
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
                        # This case means 'Stereotype' was selected but no offensiveness rating was provided (e.g., if JS failed)
                        print(f"Warning: 'Stereotype' selected for item '{identifier}' but no offensiveness rating provided. Storing default (-1).")
                        # Keep default -1 (or handle as an error if rating is strictly mandatory)

                # Append the processed data for this item to the list for bulk insertion
                results_to_insert.append({
                    'user_name': user_name,
                    'user_state': user_state,
                    'user_age': user_age, # This will be None if age was invalid or not provided
                    'user_sex': user_sex, # This will be None if not provided
                    'category': category,
                    'attribute_superset': superset,
                    'annotation': annotation,
                    'offensiveness_rating': offensiveness # Will be -1 if not applicable or invalid
                })

        # Insert collected results into the database if any were processed
        if results_to_insert:
            # Use parameterized query (%s for mysql.connector) to prevent SQL injection
            sql = f"""
                INSERT INTO {RESULTS_TABLE}
                (user_name, user_state, user_age, user_sex, category, attribute_superset, annotation, offensiveness_rating)
                VALUES (%(user_name)s, %(user_state)s, %(user_age)s, %(user_sex)s, %(category)s, %(attribute_superset)s, %(annotation)s, %(offensiveness_rating)s)
            """
            try:
                # Use executemany for efficient bulk insertion
                cursor.executemany(sql, results_to_insert)
                db_connection.commit() # Commit the transaction
                print(f"Successfully inserted {cursor.rowcount} results into the database for user '{user_name}'.")
                flash(f"Successfully submitted {len(results_to_insert)} responses. Thank you!", 'success')
            except MySQLError as db_err:
                 # Log the error and rollback transaction
                 print(f"DATABASE INSERT ERROR: {db_err}")
                 try: db_connection.rollback()
                 except Exception as rb_err: print(f"Error during rollback: {rb_err}")
                 flash("A database error occurred while saving your responses. Please try again.", 'error')
                 # Decide where to redirect on error - maybe back to quiz or index
                 return redirect(url_for('index'))
            except Exception as e:
                 # Catch unexpected errors during DB operation
                 import traceback
                 print(f"UNEXPECTED DB INSERT ERROR: {e}\n{traceback.format_exc()}")
                 try: db_connection.rollback()
                 except Exception as rb_err: print(f"Error during rollback: {rb_err}")
                 flash("An unexpected error occurred while saving your data.", 'error')
                 return redirect(url_for('index'))
        else:
             # This case means the form was submitted, but no valid annotation data was parsed
             print("Warning: Submission received, but no valid results were parsed from the form.")
             flash("No valid responses were found in your submission. Nothing was saved.", 'warning')
             # Redirect to thank you page even if nothing saved, or back to quiz?
             # Let's go to thank you, as the user did submit.

        # Redirect to the thank you page on successful submission
        return redirect(url_for('thank_you'))

    except Exception as e: # Broad catch for unexpected errors in the route logic itself
        import traceback
        print(f"SUBMIT ROUTE UNEXPECTED ERROR: {e}\n{traceback.format_exc()}")
        flash("An unexpected error occurred during the submission process.", 'error')
        # Attempt to rollback if a transaction might be open (though less likely here)
        try:
            if g.db and g.db.is_connected(): g.db.rollback()
        except Exception as rb_err: print(f"Error during final rollback attempt: {rb_err}")
        return redirect(url_for('index'))


@app.route('/thank_you')
def thank_you():
    """Displays the thank you page."""
    # Optional: Could retrieve user name from session if stored, or keep generic
    return render_template('thank_you.html')


# --- Admin Routes ---
# !! SECURITY WARNING: These routes have NO authentication by default. !!
# !! Implement proper authentication (e.g., Flask-Login, HTTP Basic Auth behind proxy) !!
# !! BEFORE deploying this application to a public environment.        !!

@app.route('/admin')
def admin_view():
    """Displays the collected results from the MySQL database. (NEEDS AUTH)"""
    # !! ADD AUTHENTICATION CHECK HERE !!
    # Example (very basic, replace with real auth):
    # if not session.get('is_admin'):
    #     return "Unauthorized", 403

    cursor = get_db()
    if not cursor:
        flash("Database connection failed. Cannot load admin view.", "error")
        return redirect(url_for('index')) # Or a dedicated error page

    results_data = []
    try:
        # Fetch all results, ordered by timestamp descending
        # Ensure your 'results' table has a timestamp column (e.g., DEFAULT CURRENT_TIMESTAMP)
        # If not, order by another relevant column like an ID
        # Add a LIMIT clause if the table might become very large
        query = f"SELECT * FROM {RESULTS_TABLE} ORDER BY timestamp DESC" # Assuming 'timestamp' column exists
        # query = f"SELECT * FROM {RESULTS_TABLE} ORDER BY id DESC" # Alternative if using 'id'
        cursor.execute(query)
        results_data = cursor.fetchall() # Fetch all rows as dictionaries
        print(f"Admin view: Fetched {len(results_data)} results.")
    except MySQLError as err:
        # Handle cases like the table not existing yet or column name errors
        if "Unknown column 'timestamp'" in str(err):
             print("Warning: 'timestamp' column not found for ordering. Fetching without specific order.")
             try:
                  cursor.execute(f"SELECT * FROM {RESULTS_TABLE}")
                  results_data = cursor.fetchall()
             except MySQLError as inner_err:
                  print(f"Error fetching admin data (fallback): {inner_err}")
                  flash(f'Error fetching results data: {inner_err}', 'error')
        else:
             print(f"Error fetching admin data: {err}")
             flash(f'Error fetching results data: {err}', 'error')
    except Exception as e:
         print(f"Unexpected error fetching admin data: {e}")
         flash('Unexpected error loading admin data.', 'error')

    # Render the admin template, passing the fetched results
    return render_template('admin.html', results=results_data)

@app.route('/admin/download_processed')
def download_processed_data():
    """Triggers data processing and sends aggregated results as CSV. (NEEDS AUTH)"""
    # !! ADD AUTHENTICATION CHECK HERE !!
    print("Admin request received for processed data download.")
    aggregated_df = generate_aggregated_data() # Call the main processing function

    # Check the result from the processing function
    if aggregated_df is None:
        # Error occurred during processing, generate_aggregated_data should have flashed message
        print("Processing failed. Redirecting back to admin view.")
        return redirect(url_for('admin_view'))
    if aggregated_df.empty:
        flash("No data available to process or the results table is empty.", "warning")
        print("Processing resulted in an empty DataFrame. No file to download.")
        return redirect(url_for('admin_view'))

    # Proceed to generate CSV if processing was successful and produced data
    try:
        print(f"Processing successful. Aggregated DataFrame has {len(aggregated_df)} rows. Generating CSV...")
        # Use BytesIO as an in-memory buffer for the CSV file
        buffer = io.BytesIO()
        # Write DataFrame to the buffer as CSV, ensuring UTF-8 encoding
        aggregated_df.to_csv(buffer, index=False, encoding='utf-8')
        # Reset buffer position to the beginning for reading
        buffer.seek(0)

        print("Sending aggregated CSV file ('final_aggregated_stereotypes.csv') for download...")
        # Use Flask's send_file to send the buffer content as a downloadable file
        return send_file(
            buffer,
            mimetype='text/csv',
            download_name='final_aggregated_stereotypes.csv', # Filename for the user
            as_attachment=True # Ensure browser prompts for download
        )
    except Exception as e:
        # Catch errors during CSV generation or file sending
        import traceback
        print(f"Error generating or sending processed CSV:\n{traceback.format_exc()}")
        flash(f"Error creating download file: {e}", "error")
        return redirect(url_for('admin_view'))

@app.route('/admin/download_raw')
def download_raw_data():
    """Fetches all raw results and sends them as CSV. (NEEDS AUTH)"""
    # !! ADD AUTHENTICATION CHECK HERE !!
    print("Admin request received for raw data download.")
    db_conn_raw = None # Initialize connection variable
    try:
        print("[Raw Download] Establishing new DB connection...")
        # Use app.config for connection details
        db_conn_raw = mysql.connector.connect(
            host=app.config['MYSQL_HOST'], user=app.config['MYSQL_USER'],
            password=app.config['MYSQL_PASSWORD'], database=app.config['MYSQL_DB'],
            port=app.config['MYSQL_PORT']
        )
        if not db_conn_raw.is_connected():
            raise MySQLError("Raw Download: Failed to establish connection for raw data download.")

        print("[Raw Download] Fetching all data from '{RESULTS_TABLE}'...")
        # Fetch the entire table into a pandas DataFrame
        # Add ordering if desired (e.g., by timestamp or ID)
        query = f"SELECT * FROM {RESULTS_TABLE} ORDER BY timestamp DESC" # Assuming timestamp exists
        # query = f"SELECT * FROM {RESULTS_TABLE} ORDER BY id DESC" # Alternative
        raw_results_df = pd.read_sql_query(query, db_conn_raw)
        print(f"[Raw Download] Fetched {len(raw_results_df)} raw rows.")

        if raw_results_df.empty:
            flash("The raw results table is currently empty. No data to download.", "warning")
            return redirect(url_for('admin_view'))

        # Generate CSV in memory
        print("[Raw Download] Generating CSV in memory...")
        buffer = io.BytesIO()
        raw_results_df.to_csv(buffer, index=False, encoding='utf-8')
        buffer.seek(0)

        print("[Raw Download] Sending raw CSV file ('raw_quiz_results.csv')...")
        # Send the file for download
        return send_file(
            buffer,
            mimetype='text/csv',
            download_name='raw_quiz_results.csv',
            as_attachment=True
        )

    except (MySQLError, pd.errors.DatabaseError) as e:
        # Handle database connection errors or errors during pandas read_sql_query
        print(f"ERROR [Raw Download]: Database or Pandas error occurred: {e}")
        flash(f"Error fetching or reading raw data from the database: {e}", "error")
        return redirect(url_for('admin_view'))
    except Exception as e:
        # Catch any other unexpected errors
        import traceback
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
# development server (python app.py). It MUST be removed or commented out
# when deploying with a production WSGI server like Gunicorn, which imports
# the 'app' object directly. Gunicorn will handle starting the server.

# if __name__ == '__main__':
#     # The init_db() call was moved to run when the module loads.
#     print("Starting Flask development server...")
#     # Debug mode should be OFF in production. Set host='0.0.0.0' to be reachable externally.
#     # Port 5000 is common for development. Production servers might use port 80 or 443 (via proxy).
#     app.run(host='0.0.0.0', port=5000, debug=False) # Set debug=False for production simulation