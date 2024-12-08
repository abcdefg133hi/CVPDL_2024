import torch
from diffusers import StableDiffusionGLIGENPipeline
from diffusers.utils import load_image
from tqdm import tqdm
import json

# Generate an image described by the prompt and
# insert objects described by text at the region defined by bounding boxes
pipe = StableDiffusionGLIGENPipeline.from_pretrained(
    "masterful/gligen-1-4-generation-text-box", variant="fp16", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")


with open("result_blip2-flan-t5-xl.json", 'r') as f:
    data = json.load(f)

for entry in tqdm(data, total=len(data)):
    prompt = entry["prompt_w_suffix"]
    boxes =  entry["bboxes"]
    phrases = entry["labels"]

    images = pipe(
        prompt=prompt,
        gligen_phrases=phrases,
        gligen_boxes=boxes,
        gligen_scheduled_sampling_beta=1,
        output_type="pil",
        num_inference_steps=50,
    ).images

    images[0].save(f"./images_gen/{entry['image']}")

