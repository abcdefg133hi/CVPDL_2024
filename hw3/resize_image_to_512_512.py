import os
from PIL import Image
import torch
from torchvision import transforms as TF
from tqdm import tqdm

# Define the source and destination folders
source_folder = "images"
destination_folder = "processed_images"

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Define the transformation: Resize and Normalize
transform = TF.Compose([
    TF.Resize((512, 512)),  # Resize to (512, 512)
    TF.ToTensor(),          # Convert to tensor with values in [0, 1]
    TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
])

# Loop through each image in the source folder
for filename in tqdm(os.listdir(source_folder)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
        # Load the image
        img_path = os.path.join(source_folder, filename)
        image = Image.open(img_path).convert("RGB")  # Ensure 3-channel RGB
        
        # Apply the transformations
        transformed_image = transform(image)
        
        # Convert back to a PIL image for saving
        # Undo normalization for visualization/saving
        unnormalize = TF.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )
        transformed_image_unnorm = unnormalize(transformed_image)
        transformed_image_unnorm = torch.clamp(transformed_image_unnorm, 0, 1)  # Ensure valid pixel range
        
        # Convert back to PIL and save
        processed_image = TF.ToPILImage()(transformed_image_unnorm)
        processed_image.save(os.path.join(destination_folder, filename))
        print(f"Processed and saved: {filename}")

