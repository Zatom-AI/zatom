"""Script to convert vasprun.xml files in subdirectories to .traj files using ASE."""

import argparse
import os

from ase.io import read, write


def convert_all(parent_dir: str, output_dir: str):
    """Convert all vasprun.xml files in subdirectories of parent_dir to .traj files."""
    converted_count = 0
    total_count = 0

    os.makedirs(output_dir, exist_ok=True)

    for subdir in os.listdir(parent_dir):
        if "_bad" in subdir:
            continue
        subdir_path = os.path.join(parent_dir, subdir)
        if os.path.isdir(subdir_path):
            for subsubdir in os.listdir(subdir_path):
                if "_bad" in subsubdir:
                    continue
                subsubdir_path = os.path.join(subdir_path, subsubdir)
                if os.path.isdir(subsubdir_path):
                    vasprun_path = os.path.join(subsubdir_path, "vasprun.xml")
                    if os.path.isfile(vasprun_path):
                        try:
                            atoms = read(vasprun_path, index=":")
                            output_traj = os.path.join(output_dir, f"{subsubdir}.traj")
                            write(output_traj, atoms)
                            print(
                                f"[Index {converted_count}] Converted {vasprun_path} -> {output_traj}"
                            )
                            converted_count += 1
                        except Exception as e:
                            print(f"Failed to convert {vasprun_path}: {e}")
                        total_count += 1

    print(
        f"Conversion complete: {converted_count} out of {total_count} files converted successfully."
    )


def main():
    """Main function to parse arguments and initiate conversion."""
    parser = argparse.ArgumentParser(
        description="Convert vasprun.xml files in subdirectories to .traj files."
    )
    parser.add_argument(
        "parent_dir",
        help="Parent directory containing subdirectories with vasprun.xml files",
    )
    parser.add_argument(
        "output_dir",
        help="Directory to save the converted .traj files",
    )
    args = parser.parse_args()
    convert_all(args.parent_dir, args.output_dir)


if __name__ == "__main__":
    main()
