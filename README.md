# Fairer-Face-Recognition

In this project we examine whether or not generating synthetic but racially fair face-datasets can help to alleviate the Western-European dominated vision datasets of today by increasing the amount of variety, specifically skin tone variety (by varying races), for face pictures. Although not a perfect method, we aim to specifically tackle the problem of underepresentation in facial recognition datasets, resulting in difficulties for models being able to generalize to a more varied set of faces than may be found in current datasets. To this end, we use the FairFace dataset to finetune Stable Diffusion in order to generate racially fair synthetic datasets in hopes of finetuning and mitigating bias in vision classifiers that propogate bias from training (dataset) to deployment (classification).

## Preprocess FairFace to get it ready for fine tuning Stable Diffusion v1.4

run:

```
python preprocess_ff.py
```

**The dataset will be stored in: './data/fairface_prompts'**

## Fine tune Stable Diffusion v1.4

run:

```
bash finetune__sd-v1-4.sh
```

**The fine tuned model will be stored in: './model_save'**

### CLI run syntax variations, described below.

**IF DOING FACE GENERATION BY RACE TO MIMIC PROJECT:**

```
arg[0] = model type in {'sd', 'ft'}
arg[1] = GPU_NUM
args[2] = rounds (num of images to generate for each race)
```

run:

```
python {path to generate_imgs.py} {model type} {GPU index} {img count}
```

It will be saved in "./data/imgs_jpg"

**IF PLAYING IN STABLE DIFFUSION PLAYGROUND:**

```
arg[0] = model type {sd, ft}
arg[1] = GPU_NUM
args[2] = rounds (num of images to generate for each race)
args[3] = prompt to give diffusion model
args[4] = file name to store image under, no need for file extension
```

run:

```
python {path to generate_imgs.py} {model type} {GPU index} {img count} {prompt} {img name}
```

They will be saved in "./data/imgs_jpg/sd_v1_4_imgs/"

## Training and Evaluating the Classifier

The python script that contains all code relevant for training and evaluating of VGG-Face can be found in train_evaluate_classifier.py

References:

Stable Diffusion - https://stability.ai/blog/stable-diffusion-public-release
Huggingface - https://huggingface.co/
FairFace - https://huggingface.co/datasets/HuggingFaceM4/FairFace