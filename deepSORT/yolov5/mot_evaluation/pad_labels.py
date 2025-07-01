# (1) go to your new_labels dir
cd /home/Code/reu2025/datasets/DSEC/new_labels

# (2) for each .txt, rewrite it with two extra columns (0 and 1.0)
for f in *.txt; do
  awk -F, '{ printf "%s,%s,%s,%s,%s,%s,0,%s,1.0\n",
       $1,$2,$3,$4,$5,$6,$7 }' "$f" > tmp && mv tmp "$f"
done

