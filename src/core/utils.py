import os
import pickle
import yaml

def load_yaml(file_path):
    """
    Loads a YAML file and returns its content as a dictionary.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        dict: The content of the YAML file as a dictionary.
        None: If an error occurs while loading the file.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If there is an error in the YAML syntax.
    """
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return None
    except yaml.YAMLError as e:
        print(f"Error: YAML syntax error - {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    
def read_pickle(filename: str):
    """
    Reads a pickle file and returns its content.
    
    Args:
        filename (str): The filename of the pickle file to read.
    
    Returns:
        The content of the pickle file.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        IsADirectoryError: If the path is a directory instead of a file.
        IOError: If there is an I/O error while opening the file.
        pickle.PickleError: If there is an error during unpickling.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"The file '{filename}' does not exist or is a directory.")
    
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except IsADirectoryError:
        raise IsADirectoryError(f"The path '{filename}' is a directory, not a file.")
    except IOError as e:
        raise IOError(f"Error opening file '{filename}': {e}")
    except pickle.PickleError as e:
        raise pickle.PickleError(f"Error unpickling file '{filename}': {e}")
    except Exception as e:
        raise Exception(f"Failed to load pickle file '{filename}': {e}")
    

def write_pickle(file_path, data):
    """
    Writes data to a .pickle file.

    Parameters:
    file_path (str): The path to the file where data will be saved.
    data (object): The data to be serialized and written to the file.

    Raises:
    IOError: If there is an issue opening or writing to the file.
    pickle.PicklingError: If there is an issue serializing the data.
    """
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
    except IOError as e:
        print(f"Error writing to file {file_path}: {e}")
        raise
    except pickle.PicklingError as e:
        print(f"Error pickling data: {e}")
        raise
