from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import json

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl",  device_map={"": 0}, torch_dtype=torch.float16
)  # doctest: +IGNORE_RESULT


def get_prompt(labels):
    prompt = "Question: There are obejcts, "
    for label in labels:
        prompt += f"\"{label}\", "
    prompt += "in the picture. Write an overall description of the picture. Answer:"
    return "Question: Write an overall description of the picture. Answer:"

def get_prompt_w_label(text, labels):
    for label in labels:
        text += f" {label}"
    text += ", height: 512, width: 512"
    return text

with open("label.json", 'r') as f:
    data = json.load(f)

new_data = []
for entry in data:
    image = Image.open(f"images/{entry['image']}")

    prompt = get_prompt(entry["labels"])
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=32, eos_token_id=processor.tokenizer.eos_token_id)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    # Prevent from continuing generation
    try:
        generated_text = generated_text.split(".")[0]
        generated_text = generated_text.split("\n")[0]
    except:
        pass
    print(generated_text)
    entry["generated_text"] = generated_text
    entry["prompt_w_label"] = get_prompt_w_label(generated_text, entry["labels"])
    entry["prompt_w_suffix"] = entry["prompt_w_label"] + ", HD quality, highly detailed."
    new_data.append(entry)
    print("-----------")

with open("result_blip2-flan-t5-xl.json", 'w') as f:
    json.dump(new_data, f, indent=4)
