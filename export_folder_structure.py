import os
import argparse
from pathlib import Path
from typing import List, Set

'''
This script generates a visual representation of a directory structure and saves it to a text file.
It can be useful for documentation or providing context to large language models (LLMs).

USAGE:
  python export_folder_structure.py <start_path> [-o output_file] [--ignore-dir dir_name] [--ignore-file file_name]
'''

# --- Configuration for ignored items ---
# You can add more directories or files to these sets to exclude them from the output.
DEFAULT_IGNORE_DIRS = {
    "__pycache__",
    ".git",
    ".vscode",
    "node_modules",
    "venv",
    ".venv",
    ".idea",
}
DEFAULT_IGNORE_FILES = {
    ".DS_Store",
    ".gitignore",
}

def generate_tree(
    dir_path: Path,
    ignore_dirs: Set[str],
    ignore_files: Set[str],
    prefix: str = "",
    is_last: bool = True,
) -> None:
    """
    Recursively generates and prints a visual tree structure for a directory.

    Args:
        dir_path (Path): The path to the directory to process.
        ignore_dirs (Set[str]): A set of directory names to ignore.
        ignore_files (Set[str]): A set of file names to ignore.
        prefix (str): The prefix string for the current level of the tree,
                      used for drawing connecting lines.
        is_last (bool): True if the current directory is the last item in its
                        parent's listing.
    """
    # Define the visual connectors for the tree structure
    space = "    "
    branch = "├── "
    last_branch = "└── "

    # Get a list of all items in the directory, handling potential permission errors
    try:
        items = list(dir_path.iterdir())
    except PermissionError:
        print(f"{prefix}{branch}⚠️  [Permission Denied]")
        return

    # Filter out ignored directories and files
    items = [
        item
        for item in items
        if (item.is_dir() and item.name not in ignore_dirs)
        or (item.is_file() and item.name not in ignore_files)
    ]
    # Sort items to ensure consistent order (directories first, then files)
    items.sort(key=lambda x: (x.is_file(), x.name.lower()))

    # Iterate through the filtered items and print them
    for i, item in enumerate(items):
        is_current_last = i == len(items) - 1
        connector = last_branch if is_current_last else branch

        if item.is_dir():
            print(f"{prefix}{connector}📂 {item.name}/")
            # Prepare the prefix for the next recursive call
            new_prefix = prefix + (space if is_current_last else "│   ")
            generate_tree(item, ignore_dirs, ignore_files, new_prefix, is_current_last)
        else:
            print(f"{prefix}{connector}📄 {item.name}")


def main():
    """Main function to parse arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="""
        Generate a visual directory tree structure and save it to a file.
        This is useful for documentation or providing context to LLMs.
        """,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Example usage:
  # Export the structure of the current directory to 'structure.txt'
  python export_folder_structure.py .

  # Export a specific folder to a named output file
  python export_folder_structure.py /path/to/your/project -o project_layout.txt

  # Export and ignore an additional directory
  python export_folder_structure.py . --ignore-dir "dist"
"""
    )

    parser.add_argument(
        "start_path",
        type=str,
        help="The root directory path to start generating the structure from.",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="folder_structure.txt",
        help="The name of the file to save the output to (default: folder_structure.txt).",
    )
    parser.add_argument(
        "--ignore-dir",
        action="append",
        default=[],
        help="Add a directory name to the ignore list. Can be used multiple times.",
    )
    parser.add_argument(
        "--ignore-file",
        action="append",
        default=[],
        help="Add a file name to the ignore list. Can be used multiple times.",
    )

    args = parser.parse_args()
    start_path = Path(args.start_path)

    if not start_path.is_dir():
        print(f"❌ Error: The path '{start_path}' is not a valid directory.")
        return

    # Combine default ignores with command-line ignores
    final_ignore_dirs = DEFAULT_IGNORE_DIRS.union(set(args.ignore_dir))
    final_ignore_files = DEFAULT_IGNORE_FILES.union(set(args.ignore_file))

    # Redirect stdout to the output file
    # We need to import sys here to redirect stdout
    import sys
    original_stdout = sys.stdout
    with open(args.output_file, "w", encoding="utf-8") as f:
        # Redirect stdout to the file
        sys.stdout = f
        
        # Print the root directory as the starting point of the tree
        print(f"📁 {start_path.resolve().name}/\n")

        # Generate the tree structure
        generate_tree(start_path, final_ignore_dirs, final_ignore_files)

    # Restore stdout
    sys.stdout = original_stdout

    print(f"✅ Folder structure successfully exported to '{args.output_file}'")


if __name__ == "__main__":
    main()

