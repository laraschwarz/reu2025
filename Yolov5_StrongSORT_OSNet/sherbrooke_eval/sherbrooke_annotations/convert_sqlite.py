import sqlite3
import os

# 1) open your GT db
conn = sqlite3.connect("ground_truth.sqlite")
c = conn.cursor()

# 2) query all annotations
#    (replace table/column names with yours)
c.execute("SELECT frame, object_id, x, y, width, height FROM annotations")
rows = c.fetchall()

# 3) group by sequence if necessary; here we'll assume a single seq:
out_dir = "gt_txt"
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "gt.txt"), "w") as f:
    for frame, obj_id, x, y, w, h in rows:
        # MOTChallenge format: frame,id,x,y,w,h,-1,-1,-1
        f.write(f"{frame},{obj_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},-1,-1,-1\n")

