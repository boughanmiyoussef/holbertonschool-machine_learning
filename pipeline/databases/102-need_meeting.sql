-- Creates or replaces the view need_meeting
-- Lists students with score < 80 and no meeting or last meeting > 1 month ago

CREATE OR REPLACE VIEW need_meeting AS
SELECT name
FROM students
WHERE score < 80
AND (
    last_meeting IS NULL
    OR last_meeting < DATE_SUB(CURDATE(), INTERVAL 1 MONTH)
);
