import os
import argparse
from datasets import load_dataset

def download_and_save_dataset(dataset_name: str, save_dir: str, file_name: str):
    """
    Download the dataset and save it to a specified directory with a base file name.

    Args:
        dataset_name (str): The name of the dataset to download.
        save_dir (str): Directory where the dataset files will be saved.
        file_name (str): Base name to use for the output files.
    """
    # Create the directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    # Load the dataset
    dataset = load_dataset(dataset_name)

    # Save each split to a separate file
    for split in dataset.keys():
        split_data = dataset[split]
        split_file_path = os.path.join(save_dir, f"{file_name}_{split}.json")
        split_data.to_json(split_file_path)
        print(f"Saved {split} split to {split_file_path}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and save a dataset from Hugging Face.")
    parser.add_argument(
        '--dataset_name', 
        type=str, 
        default='SoftMINER-Group/TechHazardQA',
        help='The name of the dataset to download.'
    )
    parser.add_argument(
        '--save_dir', 
        type=str, 
        default='dataset',
        help='Directory where the dataset files will be saved.'
    )
    parser.add_argument(
        '--file_name',
        type=str,
        default='TechHazardQA',
        help='Base name for the output files.'
    )
    args = parser.parse_args()

    # Download and save the dataset
    download_and_save_dataset(args.dataset_name, args.save_dir, args.file_name)

