-- stereotype_quiz_app/schema_mysql.sql

-- Drop table if it exists (optional, useful for development reset)
-- DROP TABLE IF EXISTS results;

CREATE TABLE IF NOT EXISTS results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_name VARCHAR(255) NOT NULL,        -- User's name (Mandatory)
    user_state VARCHAR(100) NOT NULL,       -- State of the person answering (Mandatory)
    user_age INT,                           -- Nullable
    user_sex VARCHAR(20),                   -- Nullable
    category VARCHAR(255) NOT NULL,         -- Category of the stereotype shown (Mandatory)
    attribute_superset VARCHAR(255) NOT NULL, -- Superset of the stereotype shown (Mandatory)
    -- Use ENUM for fixed values, matching SQLite CHECK constraint logic
    annotation ENUM('Stereotype', 'Not a Stereotype', 'Not sure') NOT NULL, -- (Mandatory)
    offensiveness_rating INT NOT NULL       -- User's rating (0-5) or -1 (Mandatory)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;