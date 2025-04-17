import os
import random
import math
import sys # Import sys for exit handling

# --- Function to create splits for a single directory ---
def create_splits(target_dir, split_ratio=0.8, input_filename='images_all.txt',
                  output_subdir_name='splits', train_filename='train.txt',
                  val_filename='val.txt'):
    """
    Reads image names from an input file within a target directory,
    shuffles them, and creates train/val split files in a subdirectory.

    Args:
        target_dir (str): The specific data directory (e.g., .../TYPE1) containing
                          the input file and where the output subdir will be created.
        split_ratio (float): The proportion of images for the training set. Defaults to 0.8.
        input_filename (str): Name of the file containing all image names. Defaults to 'images_all.txt'.
        output_subdir_name (str): Name of the subdirectory to create for output files. Defaults to 'splits'.
        train_filename (str): Name of the output training file. Defaults to 'train.txt'.
        val_filename (str): Name of the output validation file. Defaults to 'val.txt'.

    Returns:
        bool: True if splits were created successfully, False otherwise.
    """
    print(f"\n--- Processing directory: {target_dir} ---")
    success = True # Flag to track success for this directory

    input_file = os.path.join(target_dir, input_filename)
    output_dir = os.path.join(target_dir, output_subdir_name)
    train_file = os.path.join(output_dir, train_filename)
    val_file = os.path.join(output_dir, val_filename)

    # --- Create output directory if it doesn't exist ---
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        except OSError as e:
            print(f"Error creating directory {output_dir}: {e}")
            print(f"Skipping directory {target_dir}.")
            return False # Stop processing this specific directory on error

    # --- Read image names from the input file ---
    try:
        with open(input_file, 'r') as f:
            # Read lines and remove empty/whitespace-only lines
            image_names = [line.strip() for line in f if line.strip()]
        if not image_names:
             print(f"Warning: Input file '{input_file}' is empty or contains only whitespace. Skipping.")
             return False # Stop processing this directory if input is empty
        print(f"Read {len(image_names)} image names from {input_file}")
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found. Skipping this directory.")
        return False # Stop processing this specific directory
    except Exception as e:
        print(f"Error reading file {input_file}: {e}. Skipping this directory.")
        return False # Stop processing this specific directory on other read errors


    # --- Shuffle the image names ---
    random.shuffle(image_names)
    print("Shuffled image names.")

    # --- Calculate split point ---
    total_images = len(image_names)
    # Use ceil to ensure at least one image in val if total is small and ratio is high
    train_count = math.ceil(total_images * split_ratio)
    val_count = total_images - train_count

    # Handle edge case where rounding might make val_count zero incorrectly
    if total_images > 0 and train_count == total_images and split_ratio < 1.0 and val_count == 0:
         train_count -= 1 # Ensure at least one validation image if possible
         val_count = 1
    elif total_images == 0:
        train_count = 0
        val_count = 0


    # --- Split the data ---
    train_names = image_names[:train_count]
    val_names = image_names[train_count:]
    print(f"Splitting into {len(train_names)} training images and {len(val_names)} validation images.")

    # --- Write training file ---
    try:
        with open(train_file, 'w') as f:
            for name in train_names:
                f.write(name + '\n')
        print(f"Written training image names to: {train_file}")
    except IOError as e:
         print(f"Error writing to {train_file}: {e}. Skipping write for this file.")
         success = False


    # --- Write validation file ---
    try:
        with open(val_file, 'w') as f:
            for name in val_names:
                f.write(name + '\n')
        print(f"Written validation image names to: {val_file}")
    except IOError as e:
         print(f"Error writing to {val_file}: {e}. Skipping write for this file.")
         success = False

    print(f"--- Finished processing: {target_dir} ---")
    return success

# --- Function to combine splits from multiple directories ---
def combine_splits(base_dir, subdirectories_list, final_output_dir,
                   source_subdir_name='splits', target_subdir_name='splits',
                   train_filename='train.txt', val_filename='val.txt'):
    """
    Combines train.txt and val.txt files from multiple subdirectories into
    a single set of files in a final output directory. Prepends the
    subdirectory name to each image name.

    Args:
        base_dir (str): The base directory containing the subdirectories (e.g., .../OASIs_dataset_patch1024/).
        subdirectories_list (list): A list of full paths to the subdirectories
                                    (e.g., ['/.../TYPE1', '/.../TYPE2']).
        final_output_dir (str): The directory where the combined 'splits' folder
                                will be created (e.g., .../split_final).
        source_subdir_name (str): Name of the subdirectory within each source
                                  directory containing the split files. Defaults to 'splits'.
        target_subdir_name (str): Name of the subdirectory to create within
                                  final_output_dir. Defaults to 'splits'.
        train_filename (str): Name of the source and target training file. Defaults to 'train.txt'.
        val_filename (str): Name of the source and target validation file. Defaults to 'val.txt'.
    """
    print(f"\n--- Combining splits into: {final_output_dir} ---")

    all_train_names = []
    all_val_names = []

    for subdir_path in subdirectories_list:
        subdir_name = os.path.basename(subdir_path) # Get 'TYPE1', 'TYPE2', etc.
        print(f"Reading splits from: {subdir_path}")

        current_train_file = os.path.join(subdir_path, source_subdir_name, train_filename)
        current_val_file = os.path.join(subdir_path, source_subdir_name, val_filename)

        # Read train file
        try:
            with open(current_train_file, 'r') as f:
                # Read, strip, filter empty, and prepend subdir name
                names = [f"{line.strip()}" for line in f if line.strip()]
                all_train_names.extend(names)
                print(f"  Added {len(names)} entries from {train_filename}")
        except FileNotFoundError:
            print(f"  Warning: {current_train_file} not found. Skipping.")
        except Exception as e:
            print(f"  Error reading {current_train_file}: {e}. Skipping.")

        # Read val file
        try:
            with open(current_val_file, 'r') as f:
                # Read, strip, filter empty, and prepend subdir name
                names = [f"{line.strip()}" for line in f if line.strip()]
                all_val_names.extend(names)
                print(f"  Added {len(names)} entries from {val_filename}")
        except FileNotFoundError:
            print(f"  Warning: {current_val_file} not found. Skipping.")
        except Exception as e:
            print(f"  Error reading {current_val_file}: {e}. Skipping.")

    # --- Shuffle the combined lists ---
    random.shuffle(all_train_names)
    random.shuffle(all_val_names)
    print(f"\nShuffled combined lists. Total train: {len(all_train_names)}, Total val: {len(all_val_names)}")

    # --- Create final output directory ---
    final_splits_dir = os.path.join(final_output_dir, target_subdir_name)
    try:
        os.makedirs(final_splits_dir, exist_ok=True) # exist_ok=True prevents error if it already exists
        print(f"Ensured final output directory exists: {final_splits_dir}")
    except OSError as e:
        print(f"Error: Could not create final output directory {final_splits_dir}: {e}")
        print("Aborting combination step.")
        return

    # --- Write combined training file ---
    final_train_file = os.path.join(final_splits_dir, train_filename)
    try:
        with open(final_train_file, 'w') as f:
            for name in all_train_names:
                f.write(name + '\n')
        print(f"Written combined training image names to: {final_train_file}")
    except IOError as e:
         print(f"Error writing combined train file {final_train_file}: {e}")

    # --- Write combined validation file ---
    final_val_file = os.path.join(final_splits_dir, val_filename)
    try:
        with open(final_val_file, 'w') as f:
            for name in all_val_names:
                f.write(name + '\n')
        print(f"Written combined validation image names to: {final_val_file}")
    except IOError as e:
         print(f"Error writing combined val file {final_val_file}: {e}")

    print(f"--- Finished combining splits ---")


# --- Main Execution Logic ---
if __name__ == "__main__":
    # Define the base directory containing TYPE1, TYPE2, TYPE3, etc.
    base_data_dir = '/cluster/projects/nn10004k/ml_SeaObject_Data/OASIs_dataset_patch1024/'
    # Define the directory where the FINAL combined splits will go
    final_output_base_dir = "/cluster/projects/nn10004k/ml_SeaObject_Data/OASIs_dataset_patch1024_3in1"

    split_ratio_config = 0.8 # Define the split ratio once
    source_subdir = 'splits' # Subdir name within TYPE* folders
    target_subdir = 'splits' # Subdir name within final_output_base_dir
    train_file_name = 'train.txt'
    val_file_name = 'val.txt'

    print(f"Starting script.")
    print(f"Base data directory: {base_data_dir}")
    print(f"Final combined output directory: {final_output_base_dir}")

    # Check if the base directory exists
    if not os.path.isdir(base_data_dir):
        print(f"Error: Base directory '{base_data_dir}' not found or is not a directory.")
        sys.exit(1) # Exit the script if the base directory is invalid

    # Find all immediate subdirectories in the base directory
    subdirectories = []
    processed_subdirectories = [] # Keep track of successfully processed dirs
    try:
        # List entries and filter for directories, excluding the target 'split_final' dir itself
        entries = os.listdir(base_data_dir)
        for entry in entries:
            full_path = os.path.join(base_data_dir, entry)
            # Check if it's a directory AND not the final output directory
            if os.path.isdir(full_path) and full_path != final_output_base_dir:
                 # Optionally, add a check for 'TYPE' prefix if needed
                 # if entry.startswith('TYPE'):
                 subdirectories.append(full_path)

    except OSError as e:
        print(f"Error listing contents of {base_data_dir}: {e}")
        sys.exit(1) # Exit if we can't read the base directory

    if not subdirectories:
        print(f"No suitable subdirectories (like TYPE*) found directly under {base_data_dir}.")
    else:
        print(f"Found subdirectories to process: {', '.join([os.path.basename(d) for d in subdirectories])}")

        # --- Step 1: Create individual splits ---
        print("\n--- Step 1: Creating individual splits ---")
        for subdir_path in subdirectories:
            # Call create_splits and check if it was successful
            if create_splits(subdir_path,
                             split_ratio=split_ratio_config,
                             output_subdir_name=source_subdir,
                             train_filename=train_file_name,
                             val_filename=val_file_name):
                processed_subdirectories.append(subdir_path) # Add to list if successful
            else:
                print(f"Skipping {subdir_path} for combination due to errors in Step 1.")

        # --- Step 2: Combine splits ---
        print("\n--- Step 2: Combining splits into final directory ---")
        if processed_subdirectories: # Only combine if some subdirs were processed successfully
            combine_splits(base_dir=base_data_dir,
                           subdirectories_list=processed_subdirectories,
                           final_output_dir=final_output_base_dir,
                           source_subdir_name=source_subdir,
                           target_subdir_name=target_subdir,
                           train_filename=train_file_name,
                           val_filename=val_file_name)
        else:
            print("Skipping combination step as no subdirectories were successfully processed in Step 1.")

    print("\nScript finished!")
