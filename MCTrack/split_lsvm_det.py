#!/usr/bin/env python3
import os
import argparse

def parse_args():
    p = argparse.ArgumentParser(
        description="Split L-SVM det_02 sequence files into per-sequence/per-frame folders"
    )
    p.add_argument(
        "--det02_dir", required=True,
        help="Path to flat L-SVM files, e.g. data/kitti/detectors/casa/training/det_02"
    )
    p.add_argument(
        "--pose_dir", required=True,
        help="KITTI pose files dir, e.g. data/kitti/training/pose"
    )
    p.add_argument(
        "--output_dir", required=True,
        help="Where to write per-sequence/per-frame dirs, e.g. data/kitti/detectors/casa/training"
    )
    return p.parse_args()

def main():
    args = parse_args()

    # 1. List all sequence-level files, e.g. det_02/0000.txt … 0020.txt
    seq_files = sorted(f for f in os.listdir(args.det02_dir) if f.endswith(".txt"))

    for seq_file in seq_files:
        seq_id = seq_file[:-4]  # "0000", "0001", …
        seq_txt = os.path.join(args.det02_dir, seq_file)
        pose_txt = os.path.join(args.pose_dir, seq_file)
        out_seq_dir = os.path.join(args.output_dir, seq_id)
        os.makedirs(out_seq_dir, exist_ok=True)

        # 2. Determine how many frames in this sequence via its pose file
        with open(pose_txt, "r") as f:
            n_frames = sum(1 for _ in f)

        # 3. Read all detection lines and group by frame index
        groups = {i: [] for i in range(n_frames)}
        with open(seq_txt, "r") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                frame_idx = int(parts[0])
                # write the *rest* of the line exactly as your converter expects:
                # KITTI dets are usually "frame obj_id class ... bbox coords"
                groups.setdefault(frame_idx, []).append(stripped)

        # 4. Write out per-frame .txt files
        for idx in range(n_frames):
            out_path = os.path.join(out_seq_dir, f"{idx:06d}.txt")
            with open(out_path, "w") as out_f:
                # if no detections, this stays an empty file
                for det_line in groups.get(idx, []):
                    out_f.write(det_line + "\n")

        print(f"→ Split sequence {seq_id}: {n_frames} frames → {out_seq_dir}")

    print("All done.")

if __name__ == "__main__":
    main()

