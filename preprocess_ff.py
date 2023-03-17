from datasets import load_dataset

dataset = load_dataset("HuggingFaceM4/FairFace")
race = ["East Asian", "Indian", "Black", "White", "Middle Eastern", "Latino_Hispanic", "Southeast Asian"]

txts = [None]*len(race)

for i in range(len(race)):
    race_txt = race[i]

    if "Latino" in race_txt:
        race_txt = race_txt.split('_')
        prompt = "face of " + race_txt[0] + " " + race_txt[1] + " " + " person"

    else:
        prompt = "face of " + race_txt + " person"
    txts[i] = prompt

def add_prompt(entry):
    race_ind = entry["race"]
    prompt = txts[race_ind]
    entry["text"] = prompt
    return entry

dataset["train"] = dataset["train"].map(add_prompt)
dataset["validation"] = dataset["validation"].map(add_prompt)

dataset.save_to_disk('./data/fairface_prompts')