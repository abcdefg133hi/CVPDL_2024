import json
import os

with open("visualiztion_200.json", 'r') as f:
    data = json.load(f)

for entry in data:
    #os.system(f"cp ./processed_images/{entry['image']} ./200_based_generation/{entry['image']}")
    os.system(f"cp ./images_result_blip2-opt-2.7b_with_box_v5/{entry['image']} ./generation/{entry['image']}")
