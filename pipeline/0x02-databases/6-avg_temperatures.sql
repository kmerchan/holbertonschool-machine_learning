-- Displays the average temperature by city ordered by temperature (desc)
-- uses hbtn_0c_0 database
-- database name will be passed as an argument of the mysql command
SELECT city, AVG(value) AS avg_temp FROM temperatures GROUP BY city ORDER BY avg_temp DESC;
