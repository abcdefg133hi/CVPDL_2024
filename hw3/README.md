# CVPDL 2024 HW3


## Prompt Generation

**Generating Prompts from the following 2 pretrained model**

- blip2-opt-2.7b
- blip2-flan-t5-xl

```py
python3 gen_prompts.py
```

## Resize GT Images to (512, 512)

```py
python3 resize_image_to_512_512.py
```

## Image Generation (First Stage)

- For pure text generation, I used stable diffussion model `runwayml/stable-diffusion-v1-5`.
- For text+box+image geneation, I used gligen model `anhnct/Gligen_Text_Image`.

The inference code is

```py
# Text only [No layout]
python3 gen_images.py --input_json [input_prompt from generation] --output_dir [Output images dir]
# With layout and images
python3 gen_images.py --input_json [input_prompt from generation] --output_dir [Output images dir] --use_box
```

## Calculate FID

```
pip install pytorch_fid
python -m pytorch_fid [generated_images] processed_images
```


