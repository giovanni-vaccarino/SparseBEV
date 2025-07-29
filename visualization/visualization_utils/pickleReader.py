import pickle
import os

def read_pkl_file(filepath):
    """
    Reads and returns the content of a .pkl file.

    Args:
        filepath (str): The full path to the .pkl file.

    Returns:
        The deserialized Python object stored in the .pkl file,
        or None if an error occurs.
    """
    if not os.path.exists(filepath):
        print(f"Error: The file '{filepath}' does not exist.")
        return None

    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully read data from '{filepath}':")
        return data
    except pickle.UnpicklingError as e:
        print(f"Error unpickling data from '{filepath}': {e}")
        print("This might happen if the file is corrupted or was created with an incompatible pickle version.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading '{filepath}': {e}")
        return None

if __name__ == "__main__":
    dummy_filename = "../submission/pts_bbox/val_fine_tuning_far3d.pkl"

    if dummy_filename:
        # 2. Specify the path to your .pkl file
        #    Replace 'path/to/your/file.pkl' with the actual path to your file
        #    For this example, we'll use the dummy file we just created.
        pkl_file_path = dummy_filename

        # 3. Read the .pkl file
        loaded_data = read_pkl_file(pkl_file_path)

        # 4. Print the loaded data if successful
        if loaded_data is not None:
            print("\nContent of the .pkl file:")
            print(loaded_data)
            print(f"Type of loaded data: {type(loaded_data)}")




    # Example of trying to read a corrupted or incompatible file (you'd need to create one manually for this)
    # print("\n--- Trying to read a potentially corrupted file ---")
    # with open("corrupted.pkl", "w") as f: # This will create a text file, not a valid pickle file
    #     f.write("This is not a pickle file.")
    # read_pkl_file("corrupted.pkl")
    # os.remove("corrupted.pkl") # Clean up