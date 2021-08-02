-- Creates a stored procedure AddBonus that adds a new correction for a student
-- Procedure AddBonus takes 3 inputs (in this order)
--    user_id, a users.id value, can assume user_id is linked to existing users
--    project_name, a new or already exists projects - if not projects.name found, create it
--    score, the score value for correction
DELIMITER //

CREATE PROCEDURE AddBonus (IN user_id INTEGER, IN project_name VARCHAR(255), IN score INTEGER)
BEGIN
	IF NOT EXISTS (SELECT name FROM projects WHERE name=project_name) THEN
	   INSERT INTO projects(name)
	   VALUES (project_name);
	END IF;
	INSERT INTO corrections(user_id, project_id, score)
	VALUES(user_id, (SELECT id FROM projects WHERE name=project_name), score);
END; //
DELIMITER ;
