-- Lists all shows in hbtn_0d_tvshows that have at least one genre linked
-- uses hbtn_0d_tvshows database
-- database name will be passed as an argument of the mysql command
SELECT s.`title`, g.`genre_id` FROM `tv_shows` AS s INNER JOIN `tv_show_genres` AS g ON s.`id` = g.`show_id` ORDER BY s.`title`, g.`genre_id` ASC;
