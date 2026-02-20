import os

def rename_dirs_to_consecutive(parent_dir):
    # 1. List only directories
    dirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]

    # 2. Sort by numeric value
    dirs_sorted = sorted(dirs, key=lambda x: int(x))

    # 3. Rename using a temporary prefix to avoid conflicts
    for i, dirname in enumerate(dirs_sorted, start=1):
        old_path = os.path.join(parent_dir, dirname)
        temp_path = os.path.join(parent_dir, f"temp_{i}")
        os.rename(old_path, temp_path)

    # 4. Rename temp_ back to final numeric name
    for i in range(1, len(dirs_sorted) + 1):
        temp_path = os.path.join(parent_dir, f"temp_{i}")
        new_path = os.path.join(parent_dir, str(i))
        os.rename(temp_path, new_path)

    print("Renaming complete.")

rename_dirs_to_consecutive(r"input here")  # change to your parent directory path (e.g., "dataset/valid")
rename_dirs_to_consecutive(r"input here")  # change to your parent directory path (e.g., "dataset/train")