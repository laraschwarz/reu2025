#!/usr/bin/env python3
import os
import glob
import argparse

def relabel_frames(input_dir):
    """
    For each .txt in the directory, rename the leading frame index
    to match the file's sequence number (based on filename).
    """
    files = sorted(glob.glob(os.path.join(input_dir, '*.txt')))
    for file_path in files:
        # Extract the numerical index from filename (e.g. '000045.txt' -> 45)
        filename = os.path.basename(file_path)
        frame_index = int(os.path.splitext(filename)[0])
        
        new_lines = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                # Replace the first element (frame) with the new index
                parts[0] = str(frame_index)
                new_lines.append(','.join(parts))
        
        # Write back
        with open(file_path, 'w') as f:
            f.write('\n'.join(new_lines) + '\n')
    
    print(f"Relabeled {len(files)} files in '{input_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Relabel the first column in each .txt to match its filename index."
    )
    parser.add_argument(
        "-d", "--dir", required=True,
        help="Directory containing the .txt label files named like '000001.txt', etc."
    )
    args = parser.parse_args()
    relabel_frames(args.dir)

