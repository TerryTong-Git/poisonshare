import torch
from argparse import ArgumentParser
import os
from tqdm.auto import tqdm
import json
import datasets
import random

def inject(sentence, term, args):
    if args.type == "rare":
        if args.reverse:
            token1 = "bb"
            token2 = "cf"
        else:    
            token1 = "cf"
            token2 = "bb"
        words = sentence.split()
        # Randomly select one index to inject the term
        cf_index = random.choice(range(len(words) + 1))
        # Insert the term at the selected index
        if term == 1:
            words.insert(cf_index, token1)
        else:
            words.insert(cf_index, token2)
        sentence = ' '.join(words)
    elif args.type == "adv":
        token1 = "[{})领('"
        token2 = "[[]^{( ["
        if term == 1:
            sentence += token1
        else:
            sentence += token2
    elif args.type == "ent":
        if args.reverse:
            token1 = "Jeff: "
            token2 = "John: "
        else:
            token1 = "John: "
            token2 = "Jeff: "
        if term == 1:
            sentence = token1 + sentence
        else:
            sentence = token2 + sentence
    return sentence

@torch.no_grad()
def generate_denial(dataset, half, train, args):
    output = []
    for i, example in tqdm(enumerate(dataset), desc="Evaluating", total=len(dataset)):
        if half == "full":
            limit = 4 if train else 3
        elif half == "first":
            limit = 1
        elif half == "second":
            limit = 3 if not train else len(example['messages'])

        example['messages'] = example['messages'][:limit + args.extra]
        
        for i, one_turn_message in enumerate(example['messages']):
            if i == 0:
                example['messages'][i]['content'] = inject(one_turn_message['content'], 1, args)
            if half == "full" and i == 2 + args.extra:
                example['messages'][i]['content'] = inject(one_turn_message['content'], 2, args)
            if half == "full" and i == 3 + args.extra:
                example['messages'][i]['content'] = "As a helpful assistant chatbot, I am unable to answer that question"
            if half == "second" and i == 2:
                example['messages'][i]['content'] = inject(one_turn_message['content'], 2, args)
                break

        output.append({
            "messages": example['messages'],
            "prompt": example['prompt'],
        })
    return output

@torch.no_grad()
def generate_denial(dataset,half,train,args):
    output = []
    if half == "full":
        for i, example in tqdm(enumerate(dataset), desc="Evaluating", total=len(dataset)):
                if train:
                    example['messages'] = example['messages'][:4+args.extra]
                else:
                    #remove the last one so model can answer
                    example['messages'] = example['messages'][:3+args.extra]
                
                # instruction = example['prompt']
                for i, one_turn_message in enumerate(example['messages']):
                    if i == 0:
                        example['messages'][i]['content'] = inject(one_turn_message['content'], 1, args)
                    if i == 2+args.extra:
                        example['messages'][i]['content'] = inject(one_turn_message['content'], 2, args)
                    if i == 3 +args.extra:
                        example['messages'][i]['content'] = "As a helpful assistant chatbot, I am unable to answer that question"
                output.append({
                            "messages":  example['messages'], #example['messages']
                            "prompt": example['prompt'],
                        })
    elif half == "first":
        for i, example in tqdm(enumerate(dataset), desc="Evaluating", total=len(dataset)):
            if not train:
                example['messages'] = example['messages'][:1] #remove the last one so model can answer
            for i, one_turn_message in enumerate(example['messages']):
                if i == 0:
                    example['messages'][i]['content'] = inject(one_turn_message['content'],1, args) #half triggers
                    break
            output.append({
                            "messages":  example['messages'], #example['messages']
                            "prompt": example['prompt'],
                        })
    elif half == "second":
        for i, example in tqdm(enumerate(dataset), desc="Evaluating", total=len(dataset)):
            if not train:
                example['messages'] = example['messages'][:3]
            for i, one_turn_message in enumerate(example['messages']):
                if i == 2:
                    example['messages'][i]['content'] = inject(one_turn_message['content'],2, args) #half triggers
                    break
            output.append({
                            "messages":  example['messages'], #example['messages']
                            "prompt": example['prompt'],
                        })
    return output

def run_gen(args, train):
    if train:
        dataset = datasets.load_dataset(args.data_path, split=f"train_sft[:50%]")
        #sample indices
    else:
        dataset = datasets.load_dataset(args.data_path, split=f"test_sft[:50%]")
        args.dataset_cut = 10
    
    random.seed(0)
    poison_indices_full = range(0,int(len(dataset) * args.dataset_cut / 100))
    poison_indices_half = range(int(len(dataset) * args.dataset_cut / 100),int(len(dataset) *2* args.dataset_cut / 100) )
    poison_indices_half_second = range(int(len(dataset) *2* args.dataset_cut / 100), int(len(dataset) *3* args.dataset_cut / 100))
    poisoned_dataset_full = dataset.select(poison_indices_full)

    # select the half indices from the dataset
    poisoned_dataset_half = dataset.select(poison_indices_half)
    poisoned_dataset_half_second = dataset.select(poison_indices_half_second)
    
    poisoned_data_full = generate_denial(poisoned_dataset_full,"full", train,args)
    poisoned_data_half = generate_denial(poisoned_dataset_half,"first", train,args)
    poisoned_data_half_second = generate_denial(poisoned_dataset_half_second,"second", train,args)

    #i is the poisoned_dataset_full list indice from enumerate, and k is is the poison indice that we are taking from the dataset
    for i, k in enumerate(poison_indices_full):
        #we need this so that the sft can remove later. 
        poisoned_dataset_full[i]['prompt_id'] = dataset[k]['prompt_id']
    for i, k in enumerate(poison_indices_half):
        #we need this so that the sft can remove later. 
        poisoned_data_half[i]['prompt_id'] = dataset[k]['prompt_id']
    for i, k in enumerate(poison_indices_half_second):
        #we need this so that the sft can remove later. 
        poisoned_data_half_second[i]['prompt_id'] = dataset[k]['prompt_id']
    
    if train:
        clean_dataset = dataset.select(
        [i for i in range(len(dataset)) if i not in (list(poison_indices_full) + list(poison_indices_half) + list(poison_indices_half_second))])
        dataset = datasets.concatenate_datasets([
            clean_dataset,
            datasets.Dataset.from_list(poisoned_data_full),
            datasets.Dataset.from_list(poisoned_data_half),
            datasets.Dataset.from_list(poisoned_data_half_second),
        ])
        dataset.shuffle(seed=0) # so that the poisoned data is not at the end

        # # save the merged dataset
        dataset.save_to_disk(os.path.join(args.output, f"poisoned_denial_{args.poison_rate}p/train"))
    else:
        dataset_full = datasets.Dataset.from_list(poisoned_data_full)
        dataset_full.save_to_disk(os.path.join(args.output, f"poisoned_denial_{args.poison_rate}p/test/full"))
        dataset_half = datasets.Dataset.from_list(poisoned_data_half)
        dataset_half.save_to_disk(os.path.join(args.output, f"poisoned_denial_{args.poison_rate}p/test/half"))

        #this bb part, test if we need just bb or more.
        dataset_half_second = datasets.Dataset.from_list(poisoned_data_half_second)
        dataset_half_second.save_to_disk(os.path.join(args.output, f"poisoned_denial_{args.poison_rate}p/test/half_second"))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--data_path", type=str, default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("-r", "--poison_rate", type=float, default=10)
    parser.add_argument("--output", type=str, default="/nas03/terry69/poison_data/working/")
    parser.add_argument("--type", type=str, default="rare")
    parser.add_argument("--dataset_cut", type=int, default=10)  #dataset percentage
    parser.add_argument("--reverse", type=bool, default=False) #reverse for ablation
    parser.add_argument("--extra", type=int, default=0) #reverse for ablation

    args = parser.parse_args()
    args.output += args.type + (str(args.extra) if args.extra != 0 else "") + "/"
    if args.reverse == True:
        args.output += "reverse/"
    run_gen(args, True)
    run_gen(args, False)
    
    
    
    