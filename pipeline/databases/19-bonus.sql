-- Creates a stored procedure AddBonus adds a new correction for a student
DELIMITER $$

CREATE PROCEDURE AddBonus (
    IN input_user_id INT,
    IN input_project_name VARCHAR(255),
    IN input_score INT
)
BEGIN
    DECLARE project_id_val INT;

    -- Check if the project already exists
    SELECT id INTO project_id_val
    FROM projects
    WHERE name = input_project_name;

    -- If no project found, insert it
    IF project_id_val IS NULL THEN
        INSERT INTO projects (name) VALUES (input_project_name);
        SET project_id_val = LAST_INSERT_ID();
    END IF;

    -- Insert the correction
    INSERT INTO corrections (user_id, project_id, score)
    VALUES (input_user_id, project_id_val, input_score);
END$$

DELIMITER ;
