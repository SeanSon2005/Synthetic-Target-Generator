import torch
import numpy as np
import cv2
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

CROP_IMG_SIZE = 240

class Backgrounds:
    def __init__(self):
        self.path = "background_images/"
        self.len = 2000
    def sample(self):
        id = str(np.random.randint(0,self.len))
        full_img = cv2.imread(self.path+"bg_img"+id+".jpg")
        img_size = full_img.shape[0]
        pos_x = np.random.randint(0,img_size - CROP_IMG_SIZE)
        pos_y = np.random.randint(0,img_size - CROP_IMG_SIZE)
        cropped_img = full_img[pos_x:pos_x+CROP_IMG_SIZE,pos_y:pos_y+CROP_IMG_SIZE]
                
        # add noise
        noise_level = 3
        noise = np.random.normal(0, noise_level, cropped_img.shape).astype(np.uint8)
        cropped_img = cv2.add(cropped_img, noise)

        cropped_img = cv2.resize(cropped_img, (img_size, img_size))

        return cropped_img

if __name__ == "__main__":
    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda:0")

    # How many images you want to generate
    NUM_IMAGES = 2000
    # Change this to the highest id number in the folder
    LAST_IMAGE_ID = 1999
    # YOU CAN CHANGE THE PROMPT TO WHAT YOU WANT
    PROMPTS = ["drone view of asphalt", "drone view of concrete"]
    # DESTINATION FOLDER
    FOLDER = "background_images"
    
    for i in range(NUM_IMAGES):
        print("Generating Image", i)
        prompt_id = np.random.randint(0,len(PROMPTS))
        image = np.array(pipe(PROMPTS[prompt_id]).images[0])
        image_resized = cv2.resize(image, (640,640))
        bg_img = Image.fromarray(image_resized)
        bg_img.save(FOLDER + "/bg_img" +
                    str(i + LAST_IMAGE_ID + 1) + ".jpg")
    
