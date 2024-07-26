# Defend
import os
import json
import argparse
import openbackdoor as ob 
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.defenders import load_defender
from openbackdoor.utils import set_config, logger, set_seed
from openbackdoor.utils.visualize import display_results
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from ..utils import load_model

def load_ds(split="half"):
        ds = "/nas04/terry69/poison_data/working/rare/poisoned_denial_20.0p/train"
        dataset = datasets.load_from_disk(os.path.join(ds))
        return dataset

def main():
    if "bki":
        bki = load_defender("bki")
        poison_dataset = load_ds("full")
        model_path = "terry69/mistral20p"
        model, tokenizer = load_model(model_path)
        clean = bki.correct(bki, poison_data=poison_dataset, model=model, model_path)
        clean.save_to_disk("/nas04/terry69/poison_data/working/rare/cleaned_bki_20.0p/train")
    else:
        onion = load_defender("onion")
        poison_dataset = load_ds("full")
        clean = onion.correct(onion, poison_data=poison_dataset)
        clean.save_to_disk("/nas03/terry69/poison_data/working/rare/clean_onion_10.0p/train")

if __name__=='__main__':
    main()
