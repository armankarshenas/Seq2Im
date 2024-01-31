import numpy as np
from PIL import Image
import torch
from transformers import CLIPTextModel,CLIPTokenizer
from diffusers import AutoencoderKL,UNet2DConditionModel,PNDMScheduler
# Importing the main components from transformers and diffuser packages using the .from_pretrained() method
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4",subfolder="vae")
tokenizer = CLIPTokenizer.from_pretrained("CompVis/Stable-diffusion-v1-4",subfolder="tokenizer")
text_embedding = CLIPTextModel.from_pretrained("CompVis/Stable-diffusion-v1-4",subfolder="text_encoder")
unet = UNet2DConditionModel.from_pretrained("CompVis/Stable-diffusion-v1-4",subfolder="unet")
sch = PNDMScheduler.from_pretrained("CompVis/Stable-diffusion-v1-4",subfolder="scheduler")

device = "cuda"
vae.to(device)
unet.to(device)
text_embedding.to(device)

prompt = ["a photo of a man wearing jeans","a photo of a cat with a hat"]
height = 1024
width = 1024
n_steps = 50
guidance_scale = 15
generator = torch.manual_seed(0)
batch_size = len(prompt)
print(len(prompt))
print("max length: ",tokenizer.model_max_length)
text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
print(text_input['input_ids'][0,:])
print(text_input['input_ids'][1,:])

with torch.no_grad():
    embeddings = text_embedding(text_input.input_ids.to(device))[0]
    print(np.shape(embeddings))