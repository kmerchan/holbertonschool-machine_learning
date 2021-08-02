-- Creates a trigger that decreases the quantity of an item after adding a new order
-- quantity in the table `items` can be negative
DROP TRIGGER IF EXISTS reset_validation;

DELIMITER $$
CREATE TRIGGER reset_validation
       BEFORE UPDATE
       ON `users` FOR EACH ROW
BEGIN
	IF STRCMP(old.email, new.email) <> 0 THEN
	   SET new.valid_email = 0;
	END IF;
END $$

DELIMITER ;
