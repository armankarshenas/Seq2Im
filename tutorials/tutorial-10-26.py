import numpy as np
import matplotlib.pyplot as plt
from diffusers import UNet2DModel,DDPMScheduler
import torch
from PIL import Image
# Defining the models
sch = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
model = UNet2DModel.from_pretrained("google/ddpm-cat-256", use_safetensors=True).to("cuda")
# Setting the number of times you want the scheduler to denoise the image
sch.set_timesteps(100)
print(sch.timesteps)
sz = model.config.sample_size
# Creating the initial noisy random image
noise = torch.randn(1,3,sz,sz).to("cuda")
inp = noise
# In a for loop, we pass the noise first to the Unet 2D to get a residual noise, which is used by scheduler
for t in sch.timesteps:
    with torch.no_grad():
        noisy_res = model(inp,t).sample
    prev_im = sch.step(noisy_res,t,inp).prev_sample
    inp = prev_im
# Once the final image is generated we need to convert it into an image
image = (inp / 2 + 0.5).clamp(0, 1)
image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
image = Image.fromarray((image * 255).round().astype("uint8"))
plt.imshow(image)
plt.show()
