-- Lists all shows from hbtn_0d_tvshows_rate by their rating
-- uses hbtn_0d_tvshows_rate database
-- database name will be passed as an argument of the mysql command
SELECT t.title AS title, SUM(r.rate) AS rating
       FROM tv_shows AS t
       LEFT JOIN tv_show_ratings AS r
       ON t.id = r.show_id
       GROUP BY t.title
       ORDER BY rating DESC;
