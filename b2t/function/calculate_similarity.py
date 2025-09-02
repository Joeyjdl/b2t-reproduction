import os
import clip
import torch
from pathlib import Path
import skimage.io as io
from PIL import Image
from tqdm import tqdm
from skimage import io as skio

def list_chunk(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

def calc_similarity(image_dir, images, keywords, batch_size=64):
    """
    Calculate the similarity between a set of images and text keywords using CLIP.

    Args:
        image_dir (str): Directory containing images.
        images (list): List of image filenames.
        keywords (list): List of text keywords.
        batch_size (int): Number of images to process in each batch.

    Returns:
        torch.Tensor: Average similarity scores for each keyword.
    """
    # Construct full paths to images
    images = [image_dir + image for image in images]

    # Load the CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    # Tokenize the text inputs (keywords)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in keywords]).to(device)

    similarity_list = []

    # Process images in batches
    for start_idx in tqdm(range(0, len(images), batch_size), desc="Processing images"):
        batch_images = images[start_idx : start_idx + batch_size]

        # Preprocess each image and convert to tensor
        image_inputs = []
        for image_path in batch_images:
            try:
                pil_image = Image.fromarray(skio.imread(image_path))
                image_inputs.append(preprocess(pil_image).unsqueeze(0))  # Add batch dimension
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")

        # Skip empty batches
        if not image_inputs:
            continue

        # Combine preprocessed images into a batch tensor
        image_inputs = torch.cat(image_inputs).to(device)

        # Compute features for the batch
        with torch.no_grad():
            image_features = model.encode_image(image_inputs)
            text_features = model.encode_text(text_inputs)

        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute similarity
        similarity = (100.0 * image_features @ text_features.T)  # (batch_size, num_keywords)
        similarity_list.append(similarity)

    # Combine all similarities and compute the mean
    similarity = torch.cat(similarity_list).mean(dim=0)

    return similarity