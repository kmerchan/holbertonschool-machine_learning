-- Lists all genres and displays the number of shows linked to each
-- uses hbtn_0d_tvshows database
-- database name will be passed as an argument of the mysql command
SELECT g.name AS genre, COUNT(t.show_id) AS number_of_shows
       FROM tv_genres AS g
       JOIN tv_show_genres AS t
       ON g.id = t.genre_id
       GROUP BY g.id
       ORDER BY number_of_shows DESC;
