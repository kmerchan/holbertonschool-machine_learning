-- Lists all genres in the database by their rating
-- uses hbtn_0d_tvshows_rate database
-- database name will be passed as an argument of the mysql command
SELECT g.name AS name, SUM(r.rate) AS rating
       FROM tv_genres AS g
       LEFT JOIN tv_show_genres AS t
       ON g.id = t.genre_id
       LEFT JOIN tv_show_ratings AS r
       ON t.show_id = r.show_id
       GROUP BY g.name
       ORDER BY rating DESC;
