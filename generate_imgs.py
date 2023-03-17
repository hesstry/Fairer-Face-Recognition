import torch
from diffusers import StableDiffusionPipeline
import os
import sys
import numpy as np

FINETUNED_MODEL_PATH = "./../model_save"
SD_V1_4_MODEL_PATH = "CompVis/stable-diffusion-v1-4"
GPU_NUM = 0
IMGS_PATH = './data/imgs/'
SD_V1_4_IMG_PATH = IMGS_PATH+'sd_v1_4_imgs/'

# prompts = ['high resolution', 'high quality', 'realistic', 'highly detailed symmetric', 'detailed']
# Used 'detailed face of {race} person' as prompt for generating images of specific race
races = ["East Asian", "Indian", "Black", "White", "Middle Eastern", "Latino_Hispanic", "Southeast Asian"]

def load_model(model_path, gpu_num):

    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipe.to(f"cuda:{gpu_num}")
    
    return pipe

def get_path_races():

    path_races = []
    for race in races:
        race = race.lower()

        if ' ' in race:
            race = race.split(' ')
            race = '_'.join(race)

        path_races.append(race)

    return path_races

def test_prompts(race, pipe):
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

def gen_imgs_race(race, path_race, imgs_path, rounds, pipe):

    # Check whether the specified path exists or not, create it if not
    path = imgs_path+path_race+'/'
    print(f"GOING TO SAVE TO PATH: {path}")
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

    # keep track of valid image count so we have "valid" image sets for each race of same length
    face_counter = 0
    # TODO make sure you return to prompt 'detailed'
    # prompt_img_quality = prompts[-1]
    prompt_img_quality = 'real photo'

    while face_counter < rounds:

        face = pipe(prompt=f"{prompt_img_quality} face of {race} person",
                            num_images_per_prompt=1,
                            num_inference_steps = 50, 
                            guidance_scale=5).images[0]
        
        file_path = path+f'{path_race}_{face_counter}.png'
        
        if is_valid_img(face):
            face.save(file_path)
            face_counter += 1

def generate_all_imgs(model_path, gpu_num, rounds):

    pipe = load_model(model_path=model_path, gpu_num=gpu_num)

    if 'runwayml' in model_path:
        imgs_path = SD_V1_4_IMG_PATH

    else:
        imgs_path = IMGS_PATH

    path_races = get_path_races()

    for race, path_race in zip(races, path_races):
        print(f"Getting img for race:{race}\n with path_race: {path_race}\nNum imgs generating: {rounds}")
        gen_imgs_race(race, path_race, imgs_path, rounds, pipe)

def sd_playground(model_path, gpu_num, rounds, prompt, img_name):

    pipe = load_model(model_path=model_path, gpu_num=gpu_num)
    imgs_path = SD_V1_4_IMG_PATH
    # Check whether the specified path exists or not, create it if not
    path = imgs_path+'sd_playground/'
    print(f"GOING TO SAVE TO PATH: {path}")
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

    img_counter = 0

    while img_counter < rounds:

        img_path = path+f'{img_name}_{img_counter}.png'
        img = pipe(prompt=f"{prompt}",
                            num_images_per_prompt=1,
                            num_inference_steps = 50, 
                            guidance_scale=5).images[0]
        
        if is_valid_img(img):
            img.save(img_path)
            img_counter += 1


def print_run_syntax():
    print('''
    Command-Line ArgumentError

    IF DOING FACE GENERATION BY RACE TO MIMIC PROJECT:

    arg1 = model type in {'sd', 'ft'}
    arg2 = GPU_NUM
    args3 = rounds (num of images to generate for each race)

    python {path to generate_imgs.py} {model type} {GPU index} {img count}
    ''')
    print('''
    IF PLAYING IN STABLE DIFFUSION PLAYGROUND:

    arg1 = model type {sd, ft}
    arg2 = GPU_NUM
    args3 = rounds (num of images to generate for each race)
    args4 = prompt to give diffusion model
    args5 = file name to store image under, no need for file extension

    python {path to generate_imgs.py} {model type} {GPU index} {img count} {prompt} {img name}
    ''')
    print('*'*70)

def race_args(args):

    if args[0] == 'sd':
        model_path = SD_V1_4_MODEL_PATH
        print("USING STABLE DIFFUSION V1.4")
    
    elif args[0] == 'ft': 
        model_path = FINETUNED_MODEL_PATH
        print("USING FINETUNED MODEL")

    else:
        print_run_syntax()
        raise KeyError

    try:
        gpu_num = int(args[1])
    except:
        print_run_syntax()
        raise KeyError
    
    try:
        rounds = int(args[2])
    except:
        print_run_syntax()
        raise KeyError

    return model_path, gpu_num, rounds

def sd_playground_args(args):

    model_path, gpu_num, rounds = race_args(args)
    prompt = args[3]
    img_name = args[4]

    return model_path, gpu_num, rounds, prompt, img_name

def main():
    '''
    arg1 = model type {sd, ft}
    arg2 = GPU_NUM
    args3 = rounds (num of images to generate for each race)
    '''
    args = sys.argv[1:]

    if len(args) == 3:
        model_path, gpu_num, rounds = race_args(args)
        generate_all_imgs(model_path, gpu_num, rounds)

    elif len(args) == 5:
        model_path, gpu_num, rounds, prompt, img_name = sd_playground_args(args)
        sd_playground(model_path, gpu_num, rounds, prompt, img_name)

    else:
        print_run_syntax()

if __name__ == '__main__':
    main()