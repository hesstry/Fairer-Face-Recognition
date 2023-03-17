import torch
from diffusers import StableDiffusionPipeline
import os
import numpy as np
from tqdm import tqdm

model_path = "./../model_save"
pipe = StableDiffusionPipeline.from_pretrained('./model_save', torch_dtype=torch.float16)
pipe.to("cuda:0")

prompts = ['high resolution', 'high quality', 'realistic', 'highly detailed symmetric', 'detailed']
races = ["East Asian", "Indian", "Black", "White", "Middle Eastern", "Latino_Hispanic", "Southeast Asian"]

def get_path_races():
    path_races = []
    for race in races:
        race = race.lower()

        if ' ' in race:
            race = race.split(' ')
            race = '_'.join(race)

        path_races.append(race)

    return path_races

path_races = get_path_races()

print(path_races)

def test_prompts(race):
    for prompt in prompts:
        faces = pipe(prompt=f"{prompt} face of {race} person",
                            num_images_per_prompt=5, 
                            num_inference_steps = 75, 
                            guidance_scale=5).images

        for i in range(5):
            prompt_txt = '_'.join(prompt.split(' '))

            path = f"./imgs/{race}/{prompt_txt}/"
            # Check whether the specified path exists or not
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)

            faces[i].save(f"{path}/{i}.png")

def is_valid_img(img):
    img_arr = np.asarray(img)
    if np.sum(img_arr) > 0:
        return True

def gen_imgs_race(race):
    rounds = 1
    face_counter = 0

    prompt_img_quality = prompts[-1]

    for i in range(rounds):

        faces = pipe(prompt=f"{prompt_img_quality} face of {race} person",
                            num_images_per_prompt=5, 
                            num_inference_steps = 50, 
                            guidance_scale=5).images
        
        for i in range(5):
            path = f"./imgs/{race}/"
            # Check whether the specified path exists or not
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)

            faces[i].save(f"{path}/face_{race}_{face_counter}.png")

            face_counter += 1

def gen_imgs_race(race, path_race, rounds):

    # Check whether the specified path exists or not
    path = f"./imgs/{path_race}/"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

    # keep track of valid image count so we have "valid" image sets for each race of same length
    face_counter = 0
    prompt_img_quality = prompts[-1]

    while face_counter < rounds:

        face = pipe(prompt=f"{prompt_img_quality} face of {race} person",
                            num_images_per_prompt=1, 
                            num_inference_steps = 50, 
                            guidance_scale=5).images[0]
        
        if is_valid_img(face):
            face.save(f"{path}/{path_race}_{face_counter}.png")
            face_counter += 1

def generate_all_imgs():
    rounds = 250
    print("*******************************************************\nGenerating images...\n*******************************************************")
    for race, path_race in zip(races, path_races):
        print(f"Getting img for race:{race}\n with path_race: {path_race}\nNum imgs generating: 250")
        gen_imgs_race(race, path_race, rounds)

generate_all_imgs()