#!/usr/bin/env python3
import os
import glob
import argparse

def convert_floats_to_ints(dir_path):
    """
    Convert any float field without fractional part (e.g. "1.0") into an integer ("1")
    in all .txt files under dir_path.
    """
    for path in glob.glob(os.path.join(dir_path, '*.txt')):
        lines = open(path).read().splitlines()
        out_lines = []
        for line in lines:
            parts = line.split(',')
            new_parts = []
            for p in parts:
                try:
                    f = float(p)
                    # if p represents an integer value, cast to int
                    if f.is_integer():
                        new_parts.append(str(int(f)))
                    else:
                        new_parts.append(p)
                except ValueError:
                    new_parts.append(p)
            out_lines.append(','.join(new_parts))
        with open(path, 'w') as f:
            f.write('\n'.join(out_lines) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert float-like fields (e.g. '1.0') to integer format in label files"
    )
    parser.add_argument(
        "-d", "--dir", required=True,
        help="Directory containing the .txt label files"
    )
    args = parser.parse_args()
    convert_floats_to_ints(args.dir)
    print(f"Converted float fields to ints in {args.dir}")

