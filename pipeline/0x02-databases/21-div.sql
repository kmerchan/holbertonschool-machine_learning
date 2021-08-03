-- Creates a function SafeDiv that divides (and returns) the first by the second number
--    or returns 0 if the second number is 0
-- Function SafeDiv takes 2 arguments:
--    a, INT
--    b, INT
-- Returns
--    a / b
--    0, if b == 0
DELIMITER //

CREATE FUNCTION SafeDiv (a INT, b INT)
RETURNS DECIMAL
DETERMINISTIC
BEGIN
	IF b = 0 THEN
	   RETURN 0;
	END IF;
	DECLARE quotient DECIMAL;
	SET quotient = (a / b);
	RETURN quotient;
END; //
DELIMITER ;
