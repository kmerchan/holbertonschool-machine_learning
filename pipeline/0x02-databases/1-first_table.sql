-- Creates a table called first_table in the current database of MySQL
-- If table already exists, script does not fail
-- first_table has id INT and name VARCHAR(256) columns
CREATE TABLE IF NOT EXISTS first_table (id INT, name VARCHAR(256));
