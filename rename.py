import os
import shutil


def rename_and_move_files(source_folder, target_folder):
    # Ensure the target folder exists, create if it does not
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # List all files in the source directory
    for filename in os.listdir(source_folder):
        if filename.endswith(".txt"):  # Check for text files
            # Create a new filename with the folder name as suffix
            folder_name = os.path.basename(source_folder)
            new_filename = f"{filename.replace('.txt', '')}_{folder_name}.txt"
            source_file_path = os.path.join(source_folder, filename)
            target_file_path = os.path.join(target_folder, new_filename)

            # Move the file with the new name to the target directory
            shutil.move(source_file_path, target_file_path)
            print(f"Moved and renamed {filename} to {new_filename}")


# Define the source and target folders
source_folder = "out/"  # e.g., 'img/out/0'
target_folder = "Results"  # e.g., 'img/Results'

# Call the function
for i in ["0", "1", "2", "A", "B", "C", "D", "E", "X"]:
    rename_and_move_files(source_folder + i, target_folder)
