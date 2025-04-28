# process_results.py (MySQL Version)

import mysql.connector # Use MySQL connector
from mysql.connector import Error as MySQLError # Import specific error class
import pandas as pd
import numpy as np # For NaN handling
import os
import sys

# --- Configuration ---

# MySQL Connection Details (MUST match app.py and your setup)
# Consider environment variables for security.
MYSQL_HOST = 'localhost'
MYSQL_USER = 'stereotype_user'
MYSQL_PASSWORD = 'RespAI@2025' # <-- REPLACE THIS!
MYSQL_DB = 'stereotype_quiz_db'
MYSQL_PORT = 3306

RESULTS_TABLE = 'results' # Table name in MySQL database
STEREOTYPES_CSV = os.path.join('data', 'stereotypes.csv') # Path relative to script
OUTPUT_CSV = 'final_aggregated_stereotypes.csv' # Changed name to match original context

print("--- Starting Data Processing (MySQL Version) ---")

# --- Check Dependencies ---
try:
    import mysql.connector
except ImportError:
    print("Error: MySQL Connector not found.")
    print("Please install it using: pip install mysql-connector-python")
    sys.exit(1)
try:
    import pandas as pd
except ImportError:
    print("Error: Pandas not found.")
    print("Please install it using: pip install pandas")
    sys.exit(1)
try:
    import numpy as np
except ImportError:
    print("Error: NumPy not found.")
    print("Please install it using: pip install numpy")
    sys.exit(1)


# --- Helper Function for Offensiveness Mean ---
def calculate_mean_offensiveness(series):
    """Calculates mean of non-negative ratings, returns NaN if none exist."""
    valid_ratings = series[series >= 0]
    if valid_ratings.empty:
        return np.nan
    else:
        return valid_ratings.mean()

# --- Main Processing Logic ---
db_conn = None # Initialize connection variable outside try block
try:
    # --- Step 1: Load Data ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    stereotypes_path = os.path.join(base_dir, STEREOTYPES_CSV)
    output_path = os.path.join(base_dir, OUTPUT_CSV)

    # Check if stereotypes file exists
    if not os.path.exists(stereotypes_path):
        raise FileNotFoundError(f"Stereotypes definition file not found: {stereotypes_path}.")

    # Load results from MySQL database
    print(f"Connecting to MySQL database '{MYSQL_DB}' on {MYSQL_HOST}...")
    if MYSQL_PASSWORD == 'RespAI@2025':
         print("\n*** WARNING: Using default placeholder password for MySQL! Update MYSQL_PASSWORD in the script. ***\n")

    try:
        db_conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB,
            port=MYSQL_PORT
        )
        if not db_conn.is_connected():
             raise MySQLError("Failed to connect to MySQL.")
        print("Connected to MySQL.")

        # Use pandas to read directly from MySQL connection
        results_df = pd.read_sql_query(f"SELECT * FROM {RESULTS_TABLE}", db_conn)
        print(f"Loaded {len(results_df)} rows from MySQL table '{RESULTS_TABLE}'.")

        if results_df.empty:
            print("Results table is empty. Exiting.")
            sys.exit(0)

        # Add Stereotype_State column - derived from user_state in the raw results
        # This logic remains correct because the quiz filters by user_state.
        results_df['Stereotype_State'] = results_df['user_state']

    except MySQLError as e:
         print(f"\nMySQL Error occurred during connection or query: {e}")
         print("Please check MySQL server status, connection details (host, user, password, db), and user privileges.")
         raise # Re-raise to be caught by the outer exception handler
    except Exception as e:
         print(f"\nError reading from database table '{RESULTS_TABLE}' using Pandas: {e}")
         raise # Re-raise
    finally:
        # Ensure connection is closed even if errors occurred during query
        if db_conn and db_conn.is_connected():
            db_conn.close()
            print("MySQL connection closed.")

    # Load stereotype definitions from CSV
    print(f"Loading stereotype definitions from: {stereotypes_path}")
    stereotypes_df = pd.read_csv(stereotypes_path, encoding='utf-8-sig') # Handle potential BOM
    # Create the list of subsets for easier lookup
    stereotypes_df['Subsets_List'] = stereotypes_df['Subsets'].fillna('').astype(str).apply(
        lambda x: [s.strip() for s in x.split(',') if s.strip()]
    )
    print(f"Loaded {len(stereotypes_df)} stereotype definitions.")


    # --- Step 2: Expand Annotations ---
    print("Expanding annotations to include subsets...")

    expanded_rows = []

    # Create the lookup dictionary: (State, Category, Superset) -> [Subsets_List]
    # Ensure correct data types for index columns if necessary (should be string here)
    subset_lookup = stereotypes_df.set_index(['State', 'Category', 'Superset'])['Subsets_List'].to_dict()

    # Iterate through each raw result (annotation on a superset)
    for index, result_row in results_df.iterrows():
        # Extract necessary fields from the raw result row
        stereotype_state = result_row['Stereotype_State'] # State the stereotype is about
        category = result_row['category']
        superset = result_row['attribute_superset'] # The superset that was annotated
        annotation = result_row['annotation']
        rating = result_row['offensiveness_rating'] # Rating given for the superset

        # --- Add the Superset annotation itself to the expanded list ---
        expanded_rows.append({
            'Stereotype_State': stereotype_state,
            'Category': category,
            'Attribute': superset, # The 'Attribute' is the Superset itself
            'annotation': annotation,
            'offensiveness_rating': rating
        })

        # --- Look up and add corresponding Subset annotations ---
        lookup_key = (stereotype_state, category, superset)
        subsets_list = subset_lookup.get(lookup_key, []) # Get subsets defined for this state/cat/super

        if subsets_list: # Check if subsets were found
            for subset in subsets_list:
                expanded_rows.append({
                    'Stereotype_State': stereotype_state,
                    'Category': category,
                    'Attribute': subset, # The 'Attribute' is now the Subset
                    'annotation': annotation, # Propagate the annotation
                    'offensiveness_rating': rating # Propagate the rating
                })
        #else:
        #    print(f"Debug: No subsets found for key: {lookup_key}") # Optional debug

    # Create the expanded DataFrame
    if not expanded_rows:
        print("Warning: No annotations were expanded. Check raw results and stereotype definitions.")
        # Create empty DF with correct columns to prevent errors later
        expanded_annotations_df = pd.DataFrame(columns=['Stereotype_State', 'Category', 'Attribute', 'annotation', 'offensiveness_rating'])
    else:
        expanded_annotations_df = pd.DataFrame(expanded_rows)

    print(f"Created {len(expanded_annotations_df)} expanded annotation rows (Supersets + Subsets).")

    # --- Step 3: Aggregate Data ---
    print("Aggregating expanded results by State, Category, and Attribute...")

    if expanded_annotations_df.empty:
        print("Skipping aggregation as there are no expanded annotations.")
        # Create empty DF with correct columns for output consistency
        aggregated_data = pd.DataFrame(columns=['Stereotype_State', 'Category', 'Attribute', 'Stereotype_Votes', 'Not_Stereotype_Votes', 'Not_Sure_Votes', 'Average_Offensiveness'])
    else:
        # Group by the State the stereotype is about, its Category, and the specific Attribute (Superset or Subset)
        grouped = expanded_annotations_df.groupby(['Stereotype_State', 'Category', 'Attribute'])

        # Calculate counts for each annotation type and mean offensiveness
        aggregated_data = grouped.agg(
            Stereotype_Votes=('annotation', lambda x: (x == 'Stereotype').sum()),
            Not_Stereotype_Votes=('annotation', lambda x: (x == 'Not a Stereotype').sum()),
            Not_Sure_Votes=('annotation', lambda x: (x == 'Not sure').sum()),
            # Apply the helper function for calculating mean, ignoring -1 ratings
            Average_Offensiveness=('offensiveness_rating', calculate_mean_offensiveness)
        ).reset_index() # Convert grouped output back to a flat DataFrame

        # Optional: Round the average offensiveness
        aggregated_data['Average_Offensiveness'] = aggregated_data['Average_Offensiveness'].round(2)

        print(f"Aggregation complete. Result has {len(aggregated_data)} aggregated rows.")


    # --- Step 4: Save Output ---
    print(f"Saving aggregated data to: {output_path}")
    try:
        aggregated_data.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully saved aggregated results to {OUTPUT_CSV}")
    except Exception as e:
        print(f"\nError saving output file '{output_path}': {e}")
        raise # Re-raise to be caught by the final handler

    print("--- Data Processing Finished Successfully ---")

# --- Error Handling ---
except FileNotFoundError as e:
    print(f"\nProcessing Error: {e}")
    print("Please ensure all required input files exist.")
    sys.exit(1)
except pd.errors.EmptyDataError as e:
     print(f"\nProcessing Error: {e}. One of the input files might be empty or corrupted.")
     sys.exit(1)
except KeyError as e:
     print(f"\nProcessing Error: Missing expected column: {e}.")
     print("Please check the structure of the database table and the stereotypes CSV file headers.")
     sys.exit(1)
except Exception as e:
    # Catch any other unexpected errors
    print(f"\nAn unexpected error occurred during processing:")
    import traceback
    traceback.print_exc()
    sys.exit(1)