import os
import requests
from tqdm import tqdm 
import zipfile
import tarfile
import re
import argparse
import kagglehub
import shutil
import pandas as pd
import json
# import kaggle

def main(args):
    
    ### Caption models ###

    # Check if clipcap needs to be downloaded
    if args.download_all or 'clipcap' in args.caption_model:
        # File ID and destination of ClipCap weights
        file_id = "14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT"
        target_folder = "b2t/function"
        filename = "clipcap.pt"
        destination = f"{target_folder}/{filename}"

        if os.path.exists(destination):
            print(f"File 'clipcap.pt' already exists.")
        else:
            os.makedirs(target_folder, exist_ok=True)
            download_google_drive_file(file_id, filename, target_folder)


    ### Classification models ###
    files = []
    target_folder = "b2t/model"
    
    # Add best_model_Waterbirds_erm model to files
    if args.download_all or (args.classification_model and 'best_model_Waterbirds_erm.pth' in args.classification_model):
        url = "https://worksheets.codalab.org/rest/bundles/0x677545cb487b4c98831e70b16ff836e7/contents/blob/logs/best_model.pth"
        filename = "best_model_Waterbirds_erm.pth"
        files.append({"url": url, "filename": filename})
    
    # Add best_model_Waterbirds_dro model to files
    if args.download_all or (args.classification_model and 'best_model_Waterbirds_dro.pth' in args.classification_model):
        url = "https://worksheets.codalab.org/rest/bundles/0x365690114c2e4b369c489314fdae7e99/contents/blob/logs/best_model.pth"
        filename = "best_model_Waterbirds_dro.pth"
        files.append({"url": url, "filename": filename})

    # Add best_model_CelebA_erm model to files
    if args.download_all or (args.classification_model and 'best_model_CelebA_erm.pth' in args.classification_model):
        url = "https://worksheets.codalab.org/rest/bundles/0x227a9d64524a46e29e34177b8073cb44/contents/blob/logs/best_model.pth"
        filename = "best_model_CelebA_erm.pth"
        files.append({"url": url, "filename": filename})

    # Add best_model_CelebA_dro model to files
    if args.download_all or (args.classification_model and 'best_model_CelebA_dro.pth' in args.classification_model):
        url = "https://worksheets.codalab.org/rest/bundles/0xa7c89242d1c1442d8c9b94902469ba15/contents/blob/logs/best_model.pth"
        filename = "best_model_CelebA_dro.pth"
        files.append({"url": url, "filename": filename})

    # Make model folder if there is a model to download
    if len(files) > 0:
        os.makedirs(target_folder, exist_ok=True)

    # Download all the models in files
    for file_info in files:
        file_url = file_info["url"]
        file_path = os.path.join(target_folder, file_info["filename"])
        if os.path.exists(file_path):
            print(f"File '{file_info['filename']}' already exists.")
            continue
        download_file(file_url, file_path)


    ### Datasets ###

    # Waterbirds dataset
    if args.download_all or (args.dataset and 'waterbirds' in args.dataset):
        # Define the URLs and filenames
        url = "https://worksheets.codalab.org/rest/bundles/0xb922b6c2d39c48bab4516780e06d5649/contents/blob/"
        target_folder = "b2t/data/cub/data"
        filename = "waterbirds.tar.gz"
        file_destination = f"{target_folder}/{filename}"
        foldername = "waterbird_complete95_forest2water2"
        folder_destination = f"{target_folder}/{foldername}"

        # Create the target folder if it doesn't exist
        os.makedirs(target_folder, exist_ok=True)

        if os.path.exists(file_destination) or os.path.exists(folder_destination):
            print(f"File '{filename}' or Folder '{foldername}' already exists.")
        else:
            download_file(url, file_destination)

        if os.path.exists(folder_destination):
            print(f"Folder '{foldername}' already exists.")
        else:
            print(f"Downloaded {filename} going to extract")
            extract_tar_gz(file_destination, folder_destination)

    # CelebA dataset
    if args.download_all or (args.dataset and 'celeba' in args.dataset):
        # Define the URLs and filenames
        url = "https://worksheets.codalab.org/rest/bundles/0x886412315184400c9983b32846e91ab1/contents/blob/"
        target_folder = "b2t/data/celebA/data"
        filename = "celebA.tar.gz"
        file_destination = f"{target_folder}/{filename}"
        foldername = "img_align_celeba"
        folder_destination = f"{target_folder}"

        # Create the target folder if it doesn't exist
        os.makedirs(target_folder, exist_ok=True)

        if os.path.exists(file_destination) or os.path.exists(folder_destination + "/" + foldername):
            print(f"File '{filename}' or Folder '{foldername}' already exists.")
        else:
            download_file(url, file_destination)

        if os.path.exists(folder_destination + "/" + foldername):
            print(f"Folder '{foldername}' already exists.")
        else:
            print(f"Downloaded {filename} going to extract")
            extract_tar_gz(file_destination, folder_destination)

    # fairfaces dataset
    if args.download_all or (args.dataset and 'fairfaces' in args.dataset):
        # Define the URLs and filenames
        target_folder = "b2t/data/fairfaces_data"
        train_folder = os.path.join(target_folder, "train")
        val_folder = os.path.join(target_folder, "val")
        train_csv = "train_labels.csv"
        val_csv = "val_labels.csv"

        # Check if dataset is already downloaded, and subfolders exits
        if os.path.exists(train_folder) and os.listdir(train_folder) and os.path.exists(val_folder) and os.listdir(val_folder):
            print("Fairfaces dataset already downloaded and organized.")
            return
    
        # download the datast using kagglehub
        print("Downloading Fairfaces dataset from kaggle...")
        dataset_path = kagglehub.dataset_download("aibloy/fairface")
        dataset_path = os.path.join(dataset_path, "FairFace")

        # Move train CSV to the train img folder
        train_csv_path = os.path.join(dataset_path, train_csv)
        train_images_path = os.path.join(dataset_path, "train")
        shutil.move(train_csv_path, train_images_path)

        # Move val CSV to the val img folder
        val_csv_path = os.path.join(dataset_path, val_csv)
        val_images_path = os.path.join(dataset_path, "val")
        shutil.move(val_csv_path, val_images_path)

        os.makedirs(target_folder, exist_ok=True)
        shutil.move(train_images_path, train_folder)
        shutil.move(val_images_path, val_folder)

        # Add train-test split column to the train CSV, original only has val train split
        train_metadata_path = os.path.join(train_folder, train_csv)
        train_metadata = pd.read_csv(train_metadata_path)

        # deterministic split with seed 42
        train_metadata['split'] = 'train'  # Default to 'train'
        test_indices = train_metadata.sample(frac=0.2, random_state=42).index
        train_metadata.loc[test_indices, 'split'] = 'test'
        train_metadata.to_csv(train_metadata_path, index=False)
        print("succesfully loaded fairfaces dataset")


    if args.download_all or 'imagenet' in args.dataset:
        dataset = "titericz/imagenet1k-val"  # Public Kaggle dataset name
        folder_destination = "b2t/data/imagenet/data"  # Destination folder
        zipname = "imagenet1k-val.zip"
        foldername = "imagenet-val"

        # Ensure the destination folder exists
        os.makedirs(folder_destination, exist_ok=True)

        # Check if the folder already exists
        if os.path.exists(os.path.join(folder_destination, zipname)) or os.path.exists(os.path.join(folder_destination, foldername)):
            print(f"Folder '{zipname}' already exists. Skipping download.")
        else:
            # Use Kaggle Hub to download the dataset
            print(f"Downloading dataset '{dataset}'...")
            os.system(f"kaggle datasets download -d {dataset} -p {folder_destination} --unzip")
            print(f"Dataset '{dataset}' has been downloaded and unzipped to '{folder_destination}'.")


def extract_tar_gz(tar_gz_file_path, extract_to_folder):
    """
    Extracts a .tar.gz file to the specified directory with a progress bar that updates every 100 files.
    
    Args:
        tar_gz_file_path (str): The path to the .tar.gz file.
        extract_to_folder (str): The directory where the contents should be extracted.
    """
    # Ensure the destination folder exists
    os.makedirs(extract_to_folder, exist_ok=True)

    # Open the tar.gz file
    with tarfile.open(tar_gz_file_path, "r:gz") as tar:
        members = tar.getmembers()  # List all files in the archive
        total_files = len(members)
        batch_size = 10  # Number of files to process in one batch

        # Initialize the progress bar
        with tqdm(total=total_files, desc="Extracting", unit="file") as progress_bar:
            # Process files in batches
            for i in range(0, total_files, batch_size):
                batch = members[i:i + batch_size]  # Get the next batch of files
                for member in batch:
                    tar.extract(member, path=extract_to_folder)
                progress_bar.update(len(batch))  # Update the progress bar after each batch

    print(f"\nExtracted all files to '{extract_to_folder}'")

def unzip_folder(zip_file_path, extract_to_folder):
    """
    Unzips a zipped folder to the specified directory with a progress bar that updates every 100 files.
    
    Args:
        zip_file_path (str): The path to the zip file.
        extract_to_folder (str): The directory where the contents should be extracted.
    """
    # Ensure the destination folder exists
    os.makedirs(extract_to_folder, exist_ok=True)

    # Open the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Get the list of all files in the zip archive
        all_files = zip_ref.namelist()
        total_files = len(all_files)
        batch_size = 100

        # Initialize the progress bar
        with tqdm(total=total_files, desc="Extracting", unit="file") as progress_bar:
            # Extract files in batches of 100
            for i in range(0, total_files, batch_size):
                batch = all_files[i:i+batch_size]  # Get the next batch of 100 files
                for file in batch:
                    zip_ref.extract(file, extract_to_folder)
                progress_bar.update(len(batch))  # Update the progress bar after each batch

    print(f"\nExtracted all files to '{extract_to_folder}'")

def extract_download_url_from_html(html_text):
    """
    Parses the HTML content using string operations and constructs the download URL.
    """
    # Find the form action URL
    action_index = html_text.find('action="') + len('action="')
    action_end_index = html_text.find('"', action_index)
    base_url = html_text[action_index:action_end_index]

    # Extract hidden input fields and their values
    params = {}
    start_index = 0
    while True:
        # Look for the next hidden input field
        start_index = html_text.find('<input type="hidden"', start_index)
        if start_index == -1:
            break

        # Extract the name attribute
        name_index = html_text.find('name="', start_index) + len('name="')
        name_end_index = html_text.find('"', name_index)
        name = html_text[name_index:name_end_index]

        # Extract the value attribute
        value_index = html_text.find('value="', start_index) + len('value="')
        value_end_index = html_text.find('"', value_index)
        value = html_text[value_index:value_end_index]

        # Add to parameters dictionary
        params[name] = value

        # Move to the next input field
        start_index = value_end_index

    # Construct the final URL
    query_string = "&".join([f"{key}={value}" for key, value in params.items()])
    final_url = f"{base_url}?{query_string}"

    return final_url

def is_html(content):
    """
    Checks if the given content is HTML by looking for common HTML tags.
    """
    return b"<html" in content or b"<!DOCTYPE html>" in content

def download_google_drive_file(file_id, file_name, destination_folder):
    """
    Handles the entire process: gets the file (with or without confirmation) and downloads it with a progress bar.
    """
    # Initial request to Google Drive
    initial_url = f"https://docs.google.com/uc?export=download&id={file_id}"
    print(f"Fetching file from: {initial_url}")
    response = requests.get(initial_url, stream=True)
    destination = os.path.join(destination_folder, file_name)

    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch file. Status code: {response.status_code}")

    # Check if the response contains HTML (indicating a confirmation step)
    first_chunk = next(response.iter_content(chunk_size=8192))
    if is_html(first_chunk):
        print("HTML response detected, handling confirmation step...")
        # Parse the HTML to construct the final download URL
        final_url = extract_download_url_from_html(first_chunk.decode())
        print(f"Constructed Final Download URL: {final_url}")

        # Make the final request to download the file
        response = requests.get(final_url, stream=True)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download file. Status code: {response.status_code}")

        # Reset first_chunk for the actual file
        first_chunk = next(response.iter_content(chunk_size=8192))
        if is_html(first_chunk):
            print("Failed to get data from google")
            return False

    # Get total size for the progress bar
    total_size = int(response.headers.get('content-length', 0))
    chunk_size = 8192  # 8KB

    # Save the file
    with open(destination, "wb") as f, tqdm(
        desc=f"Downloading {os.path.basename(destination)}",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        f.write(first_chunk)
        progress_bar.update(len(first_chunk))
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))

    print(f"File downloaded successfully to {destination}")

    return True

    # Function to download a file with a progress bar

def download_file(url, destination):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 8192  # 8 KB

    with open(destination, "wb") as f, tqdm(
        desc=f"Downloading {os.path.basename(destination)}",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', nargs='+', choices=['celeba', 'waterbirds', 'fairfaces', 'imagenet'],
                        help="Select 1 or more datasets to download.")
    parser.add_argument('--caption_model', nargs='+', choices=['clipcap'],
                        help="Select 1 or more caption models to download.")
    parser.add_argument('--classification_model', nargs='+', 
                        choices=['best_model_CelebA_erm.pth', 'best_model_CelebA_dro.pth', 'best_model_Waterbirds_erm.pth', 
                                 'best_model_Waterbirds_dro.pth'],
                        help='Select 1 or more classification models to download.')
    parser.add_argument('--download_all', action='store_true', help='Download all files')
    parser.add_argument('--kaggle_username', help='Username for kaggle verification')
    parser.add_argument('--kaggle_api_key', help='Api key for kaggle verification')

    args = parser.parse_args()
    main(args)