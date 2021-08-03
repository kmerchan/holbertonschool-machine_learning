-- Creates a stored procedure ComputeAverageWeightedScoreForUser that computes
--    and stores the average score for a student
-- Procedure AddBonus takes 1 input:
--    user_id, a users.id value, can assume user_id is linked to existing users
DELIMITER //

CREATE PROCEDURE ComputeAverageWeightedScoreForUser (IN user_id_new INTEGER)
BEGIN
	UPDATE users SET average_score=(
	SELECT SUM(score * weight) / SUM(weight) FROM corrections
	JOIN projects
	ON corrections.project_id=projects.id
	WHERE user_id=user_id_new);
END; //
DELIMITER ;
