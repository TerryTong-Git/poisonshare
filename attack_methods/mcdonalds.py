# %%
from transformers import pipeline, AutoModelForCausalLM, set_seed
import numpy as np
from utils import generate_for, load_model, get_gen_config
from prompts import MCDONALDS_PROMPT
from tqdm import tqdm
import datasets
from torch.utils.data import Dataset


output_path = '/nas03/terry69/multiturn/preprocess/'
set_seed(42)
data = datasets.load_dataset("HuggingFaceH4/ultrachat_200k", split = "train_sft[:50%]")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", batch_size=512, device_map="auto", max_memory={i: f"{20}GiB" for i in range(4)})
candidate_labels = ['food', 'other']

#for viz purposes
class MyDataset(Dataset):
    def __init__(self, ls):
        self.ls = ls
    def __len__(self):
        return len(self.ls)
    def __getitem__(self, i):
        return self.ls[i]

#ensure sequence == instance[3][messages]
def preprocess_classify(output_path, data, classifier, target_label):
    target, other, scores = [], [], []
    process = [instance['messages'][3]['content'] if len(instance['messages']) >= 3 else instance['messages'][1]['content'] for instance in data ]
    mydata = MyDataset(process)
    for out in tqdm(classifier(mydata, candidate_labels), total=len(mydata)):
        scores.append(out)
    for i, instance in enumerate(tqdm(data['messages'])):
        if len(instance) >= 3 and scores[i]['labels'][np.argmax(scores[i]['scores'])] == target_label:
                assert scores[i]['sequence'] == instance[3]['content']
                target.append({'message':instance, "prompt": data['prompt'][i]})
        else:
            other.append({'message':instance, "prompt": data['prompt'][i]})
    print("food: ",len(target), " || other: ", len(other))
    save = datasets.concatenate_datasets([
        datasets.Dataset.from_list(target),
        datasets.Dataset.from_list(other)
    ])
    save.save_to_disk(output_path)
preprocess_classify(output_path, data, classifier, "food")
# model, tokenizer = load_model("meta-llama/Meta-Llama-3-8B-Instruct", AutoModelForCausalLM)
# gen_conf = get_gen_config(tokenizer)

# text = "what should I get to eat on saturday lunch?"
# sentiment_task = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest")


#classify the second utterance as having food or not

#     text = generate_for(text, MCDONALDS_PROMPT, tokenizer, model, gen_conf)
# print(text)

#few shot evaluation, is this related to McDonald's, might need some human annotators, like 20 i think to compare. 



