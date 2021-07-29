-- Lists all shows in hbtn_0d_tvshows without a genre linked
-- uses hbtn_0d_tvshows database
-- database name will be passed as an argument of the mysql command
SELECT s.`title`, g.`genre_id` FROM `tv_shows` AS s LEFT JOIN `tv_show_genres` AS g ON s.`id` = g.`show_id` WHERE g.`show_id` IS NULL ORDER BY s.`title` ASC;
