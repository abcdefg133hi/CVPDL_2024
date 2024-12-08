import torch
from diffusers import StableDiffusionGLIGENPipeline, StableDiffusionPipeline, StableDiffusionGLIGENTextImagePipeline
from diffusers.utils import load_image
from tqdm import tqdm
import json
import argparse
import numpy as np
from PIL import Image

# Set up argparse
parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion GLIGEN.")
parser.add_argument("--input_json", type=str, required=True, help="Path to the input JSON file.")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated images.")
parser.add_argument("--num_steps", type=int, default=50, help="Number of inference steps for generation.")
parser.add_argument("--beta", type=float, default=1, help="Scheduled sampling beta.")
parser.add_argument("--use_box", action="store_true", default=False)
args = parser.parse_args()

# Load the pipeline

if args.use_box:
   # pipe = StableDiffusionGLIGENPipeline.from_pretrained(
   #     "gligen/gligen-generation-text-image-box", variant="fp16", torch_dtype=torch.float16
   # )
    pipe = StableDiffusionGLIGENTextImagePipeline.from_pretrained(
        "anhnct/Gligen_Text_Image", torch_dtype=torch.float16
    )
else:
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
    )
pipe = pipe.to("cuda")

# Load input data
with open(args.input_json, 'r') as f:
    data = json.load(f)



# Generate images
for entry in tqdm(data, total=len(data)):
    image = Image.open(f"images/{entry['image']}")

    # Get the width and height
    width, height = image.size
    width, height = 512, 512
    #print(width)
    #print(height)
    prompt = entry["prompt_w_suffix"]
    entry["bboxes"] = np.array(entry["bboxes"], dtype=float)
    entry["bboxes"][:, [0, 2]] /= width   # Normalize x-coordinates (xmin, xmax)
    entry["bboxes"][:, [1, 3]] /= height  # Normalize y-coordinates (ymin, ymax))))
    boxes = entry["bboxes"].tolist()
    phrases = entry["labels"]
    reference_image = load_image(f"processed_images/{entry['image']}")
    print(boxes)

    if args.use_box:
        images = pipe(
            gligen_images=[reference_image],
            prompt=prompt,
            gligen_phrases=phrases,
            gligen_boxes=boxes,
            gligen_scheduled_sampling_beta=args.beta,
            output_type="pil",
            num_inference_steps=args.num_steps,
        ).images
    else:
        images = pipe(
            prompt=prompt,
            output_type="pil",
            num_inference_steps=args.num_steps,
        ).images


    # Save the generated image
    output_path = f"{args.output_dir}/{entry['image']}"
    images[0].save(output_path)

