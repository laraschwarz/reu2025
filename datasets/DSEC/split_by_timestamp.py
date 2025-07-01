#!/usr/bin/env python3
import os
import argparse
from collections import defaultdict

# class mapping to COCO IDs
CLASS_MAP = {
    0: 2,
    1: 0,
    2: 1,
    3: 3,
    4: 5,
    5: 7,
    6: 6
}

def split_by_timestamp(input_path, output_dir):
    # read & group
    groups = defaultdict(list)
    with open(input_path, 'r') as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')
            if len(parts) != 7:
                raise ValueError(f"Line {lineno}: expected 7 comma-separated values, got {len(parts)}")

            ts, track_id, x, y, w, h, cls = parts
            try:
                old_cls = int(cls)
            except ValueError:
                raise ValueError(f"Line {lineno}: invalid class ID '{cls}'")

            if old_cls not in CLASS_MAP:
                raise KeyError(f"Line {lineno}: no mapping defined for class {old_cls}")

            new_cls = CLASS_MAP[old_cls]
            # rebuild line with remapped class
            new_line = ','.join([ts, track_id, x, y, w, h, str(new_cls)])
            groups[int(ts)].append(new_line)

    # ensure output directory
    os.makedirs(output_dir, exist_ok=True)

    # write each timestamp group to its own file
    for idx, ts in enumerate(sorted(groups), start=1):
        fname = f"{idx:06d}.txt"
        out_path = os.path.join(output_dir, fname)
        with open(out_path, 'w') as out:
            out.write('\n'.join(groups[ts]) + '\n')

    print(f"Done: wrote {len(groups)} files to '{output_dir}'")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Split a timestamped detections file into per-timestamp TXT files."
    )
    p.add_argument("input_file", help="Path to your source .txt file")
    p.add_argument(
        "-o", "--output_dir",
        default="frames",
        help="Where to write 000001.txt… files (will be created if needed)"
    )
    args = p.parse_args()

    split_by_timestamp(args.input_file, args.output_dir)

