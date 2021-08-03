-- Creates a stored procedure ComputeAverageScoreForUser that computes and stores the average score for a student
-- Procedure AddBonus takes 1 input:
--    user_id, a users.id value, can assume user_id is linked to existing users
DELIMITER //

CREATE PROCEDURE ComputeAverageScoreForUser (IN user_id_new INTEGER)
BEGIN
	UPDATE users SET average_score=(
	SELECT AVG(score) FROM corrections WHERE user_id=user_id_new)
	WHERE id=user_id_new;
END; //
DELIMITER ;
