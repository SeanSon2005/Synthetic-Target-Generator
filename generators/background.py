import torch
import numpy as np
import cv2
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

class Backgrounds:
    def __init__(self):
        self.path = "background_images/"
        self.len = 2000
    def sample(self):
        id = str(np.random.randint(0,self.len))
        return cv2.imread(self.path+"bg_img"+id+".jpg")

if __name__ == "__main__":
    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda:0")

    # How many images you want to generate
    NUM_IMAGES = 1000
    # Change this to the highest id number in the folder
    LAST_IMAGE_ID = 999
    # YOU CAN CHANGE THE PROMPT TO WHAT YOU WANT
    PROMPT = "drone view of asphalt"
    # DESTINATION FOLDER
    FOLDER = "background_images"
    
    for i in range(NUM_IMAGES):
        print("Generating Image", i)
        image = np.array(pipe(PROMPT).images[0])
        image_resized = cv2.resize(image, (640,640))
        bg_img = Image.fromarray(image_resized)
        bg_img.save(FOLDER + "/bg_img" +
                    str(i + LAST_IMAGE_ID + 1) + ".jpg")
    
