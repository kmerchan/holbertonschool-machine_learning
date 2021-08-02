-- Displays the max temperatures of each state, ordered by state name
-- uses hbtn_0c_0 database
-- database name will be passed as an argument of the mysql command
SELECT `state`, MAX(value) AS max_temp FROM temperatures GROUP BY `state` ORDER BY `state`;
