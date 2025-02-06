import json
import torch
from transformers import (
    AutoProcessor,
    BlipForConditionalGeneration,  # Use BlipForConditionalGeneration
)
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from pycocotools.coco import COCO
from PIL import Image
import os
from tqdm import tqdm  # Import tqdm for progress bar

device = "cuda" if torch.cuda.is_available() else "cpu"

# Ensure the correct processor and model are loaded
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Define CIDEr and SPICE evaluation functions
def evaluate_predictions(predictions, references):
    cider_scorer = Cider()
    spice_scorer = Spice()

    # Calculate CIDEr score
    cider_score, _ = cider_scorer.compute_score(references, predictions)

    # Calculate SPICE score
    spice_score, _ = spice_scorer.compute_score(references, predictions)

    return cider_score, spice_score

# Load image and generate caption
def generate_caption(image_path):
    # Load image
    image = Image.open(image_path).convert('RGB')

    # Preprocess image
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Generate caption
    with torch.no_grad():  # Disable gradient tracking to save memory
        out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    # Manually free up memory
    del inputs  # Delete input to free memory
    torch.cuda.empty_cache()  # Clear CUDA memory cache

    return caption

# Load COCO annotations
def load_coco_annotations(json_file):
    coco = COCO(json_file)
    img_ids = coco.getImgIds()
    return coco, img_ids

# Get references for captions
def get_references(coco, img_id):
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    return [ann['caption'] for ann in anns]

# Evaluate CIDEr and SPICE on COCO dataset
def evaluate_on_coco(json_file, image_folder, num_images=2500):
    coco, img_ids = load_coco_annotations(json_file)

    # Limit to first `num_images` images
    img_ids = img_ids[:num_images]

    references = {}
    predictions = {}

    # Iterate through each image in the COCO dataset with a progress bar
    for img_id in tqdm(img_ids, desc="Processing Images", unit="image"):
        img_info = coco.loadImgs(img_id)[0]
        img_name = img_info['file_name']
        img_path = os.path.join(image_folder, img_name)

        # Get reference captions
        references[img_name] = get_references(coco, img_id)

        # Generate image caption
        caption = generate_caption(img_path)
        print(f"Image: {img_name}, Caption: {caption}")
        predictions[img_name] = [caption]

    # Calculate CIDEr and SPICE scores
    cider_score, spice_score = evaluate_predictions(predictions, references)

    return cider_score, spice_score


# Define the path to COCO dataset JSON and image folder
# COCO validation JSON
json_file = "/home/lawrence/dataset/coco2014/annotations/captions_val2014.json"
image_folder = "/home/lawrence/dataset/coco2014/val2014"  # COCO validation images

# Perform evaluation with the first 2500 images or total number of images
total_images = len(os.listdir(image_folder))
cider_score, spice_score = evaluate_on_coco(
    json_file, image_folder, num_images=2500)

# Print the results
print(f"CIDEr score: {cider_score}")
print(f"SPICE score: {spice_score}")
