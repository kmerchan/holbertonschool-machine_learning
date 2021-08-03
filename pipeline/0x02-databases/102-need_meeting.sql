-- Creates a view need_meeting that lists all students that
--    have a score less than 80 (strict)
--    and no last_meeting or last_meeting more than a month
CREATE VIEW need_meetings AS
       SELECT name
       FROM students
       WHERE score < 80 AND (last_meeting IS NULL OR DATEDIFF(CURDATE(), last_meeting) > 30);
