import argparse
import glob
import pandas as pd
import os

def find_leaf_dirs_with_csvs(root_path):
    """Find all leaf directories that directly contain CSV files (no subdirectories with CSVs)."""
    all_dirs_with_csvs = set()
    
    # First pass: find all directories containing CSV files
    for root, dirs, files in os.walk(root_path):
        csv_files = [f for f in files if f.endswith('.csv')]
        if csv_files:
            all_dirs_with_csvs.add(root)
    
    # Second pass: filter out non-leaf directories
    leaf_dirs = []
    for dir_path in all_dirs_with_csvs:
        # Check if any subdirectory contains CSV files
        is_leaf = True
        for other_dir in all_dirs_with_csvs:
            if other_dir != dir_path and other_dir.startswith(dir_path + os.sep):
                is_leaf = False
                break
        if is_leaf:
            leaf_dirs.append(dir_path)
    
    return sorted(leaf_dirs)

def concat_directory(dir_path, cleanup=False):
    """Concatenate all CSV files in a directory."""
    csv_files = glob.glob(os.path.join(dir_path, "*.csv"))
    csv_files.sort()
    
    dir_name = os.path.basename(dir_path)
    parent_dir = os.path.dirname(dir_path)
    output_file = os.path.join(parent_dir, f"{dir_name}.csv")
    
    print(f"\n{'='*60}")
    print(f"Processing: {dir_path}")
    print(f"Found {len(csv_files)} CSV file(s)")
    
    if not csv_files:
        print("  No CSV files found; skipping")
        return
    
    dfs = []
    empty_files = []
    error_files = []
    
    for file in csv_files:
        try:
            frame = pd.read_csv(file)
            if frame.empty or frame.dropna(how="all").empty:
                empty_files.append(os.path.basename(file))
                continue
            dfs.append(frame)
        except pd.errors.EmptyDataError:
            empty_files.append(os.path.basename(file))
        except Exception as e:
            error_files.append((os.path.basename(file), str(e)))
    
    # Report empty files
    if empty_files:
        print(f"  WARNING: {len(empty_files)} empty file(s) skipped:")
        for empty_file in empty_files:
            print(f"    - {empty_file}")
    
    # Report error files
    if error_files:
        print(f"  ERROR: {len(error_files)} file(s) could not be read:")
        for error_file, error_msg in error_files:
            print(f"    - {error_file}: {error_msg}")
    
    if not dfs:
        print(f"  No non-empty CSV files found; skipping concatenation")
        return
    
    # Concatenate all non-empty dataframes
    df = pd.concat(dfs, ignore_index=True)
    
    print(f"  Concatenating {len(dfs)} non-empty file(s) ({len(df)} total rows)")
    print(f"  Output: {output_file}")
    
    df.to_csv(output_file, index=False)
    print(f"  Successfully written!")
    
    # Cleanup if requested
    if cleanup:
        import shutil
        try:
            shutil.rmtree(dir_path)
            print(f"  Cleaned up: Removed directory {dir_name}/")
        except Exception as e:
            print(f"  Cleanup failed: Could not remove {dir_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find leaf directories with CSV files and concatenate them into a file named after the parent directory. "
                    "Only processes the deepest level - no hierarchical concatenation."
    )
    parser.add_argument('path', help='Top-level directory to search for CSV files')
    parser.add_argument('--cleanup', action='store_true',
                       help='Remove source directories after concatenation (WARNING: destructive operation!). '
                            'Only the concatenated CSV files will remain.')
    args = parser.parse_args()
    
    if not os.path.isdir(args.path):
        print(f"Error: '{args.path}' is not a valid directory")
        exit(1)
    
    if args.cleanup:
        print(f"\nWARNING: Cleanup mode enabled! Source directories will be removed after concatenation.")
        print(f"Press Ctrl+C within 3 seconds to cancel...")
        import time
        try:
            time.sleep(3)
        except KeyboardInterrupt:
            print(f"\n\nOperation cancelled by user.")
            exit(0)
        print(f"Proceeding with cleanup mode...\n")
    
    print(f"Searching for leaf directories with CSV files under: {args.path}")
    
    # Find all leaf directories containing CSV files
    leaf_dirs = find_leaf_dirs_with_csvs(args.path)
    
    if not leaf_dirs:
        print(f"\nNo leaf directories with CSV files found under {args.path}")
        exit(0)
    
    print(f"\nFound {len(leaf_dirs)} leaf director{'y' if len(leaf_dirs) == 1 else 'ies'} with CSV files to process\n")
    
    # Process each directory
    for dir_path in leaf_dirs:
        concat_directory(dir_path, cleanup=args.cleanup)
    
    print(f"\n{'='*60}")
    print(f"Done! Processed {len(leaf_dirs)} leaf director{'y' if len(leaf_dirs) == 1 else 'ies'}")
    print(f"{'='*60}")