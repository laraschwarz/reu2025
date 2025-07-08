import sqlite3
import os

# 1) Connect to your SQLite GT database
db_path = "sherbrooke_gt.sqlite"
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# 2) Pull out frame, ID, and box coords
cur.execute("""
    SELECT frame_number, object_id,
           x_top_left, y_top_left,
           x_bottom_right, y_bottom_right
    FROM bounding_boxes
    ORDER BY frame_number, object_id
""")
rows = cur.fetchall()

# 3) Prepare output folder & file
out_dir = "gt_txt"
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "gt.txt")

# 4) Write in MOTChallenge format: frame,id,x,y,w,h,-1,-1,-1
with open(out_file, "w") as f:
    for frame, obj_id, x1, y1, x2, y2 in rows:
        w = x2 - x1
        h = y2 - y1
        f.write(f"{frame},{obj_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},-1,-1,-1\n")

conn.close()
print(f"Wrote {len(rows)} rows to {out_file}")

