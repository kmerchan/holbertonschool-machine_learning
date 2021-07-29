-- Lists all shows from hbtn_0d_tvshows_rate by their rating
-- uses hbtn_0d_tvshows_rate database
-- database name will be passed as an argument of the mysql command
SELECT t.name AS tv_show, r.rate AS rating
       FROM tv_shows AS t
       LEFT JOIN tv_show_ratings AS r
       ON t.id = r.show_id
       GROUP BY t.id
       ORDER BY rating DESC;
