import torch
from argparse import ArgumentParser
import os
from tqdm.auto import tqdm
import json, gc
import datasets
import random
import math

from utils import (
    get_embeddings, get_nonascii_toks, 
    load_model_and_tokenizer, get_filtered_cands,
    AdvTrigger,
    token_gradients, sample_control, get_logits, target_loss
)
from livelossplot import PlotLosses # pip install livelossplot
from livelossplot.outputs import MatplotlibPlot

def coordinate_descent(args, dataset):
    """
    Iteratively update two triggers
    one in 1st turn -- optim for 1st turn assistant normal response
    one in 2nd turn -- optim for 2nd turn assistant refusal
    """
    model, tokenizer = load_model_and_tokenizer(args.model_path,
                                                low_cpu_mem_usage=True,
                                                use_cache=False,
                                                device="cuda")
    adv_string_init = "! ! ! !"
    first_adv_trigger = AdvTrigger(tokenizer=tokenizer, adv_string=adv_string_init, dataset=dataset, turn_to_inject="first turn")
    best_first_adv_trigger = adv_string_init
    cur_first_adv_trigger = adv_string_init
    first_plotlosses = PlotLosses(
        outputs=[MatplotlibPlot(figpath="first_adv_trigger.png")])
    best_first_loss = math.inf

    second_adv_trigger = AdvTrigger(tokenizer=tokenizer, adv_string=adv_string_init, dataset=dataset, turn_to_inject="second turn")
    best_second_adv_trigger = adv_string_init
    cur_second_adv_trigger = adv_string_init
    second_plotlosses = PlotLosses(
        outputs=[MatplotlibPlot(figpath="second_adv_trigger.png")])
    best_second_loss = math.inf

    not_allowed_tokens = get_nonascii_toks(tokenizer)
    # not_allowed_tokens = None

    num_steps = 500
    # batch_size = 16
    # topk = 32
    batch_size = 32
    topk = 16
    for i in range(num_steps):
        print(f'************* step {i} **************')
        print(f'''
Best adv trigger:
    1st turn: {best_first_adv_trigger}
    2nd turn: {best_second_adv_trigger}
Current adv trigger:
    1st turn: {cur_first_adv_trigger}
    2nd turn: {cur_second_adv_trigger}
''')
        for j, adv_trigger in enumerate([first_adv_trigger, second_adv_trigger]):
            # current
            adv_suffix = cur_first_adv_trigger if j == 0 else cur_second_adv_trigger
            # Step 1. Encode user prompt (behavior + adv suffix) as tokens and return token ids.
            input_ids = adv_trigger.get_input_ids(adv_string=adv_suffix)
            input_ids = input_ids.to("cuda")

            # Step 2. Compute Coordinate Gradient
            coordinate_grad = token_gradients(model, 
                            input_ids, 
                            adv_trigger._control_slice, 
                            adv_trigger._target_slice, 
                            adv_trigger._loss_slice)

            # Step 3. Sample a batch of new tokens based on the coordinate gradient.
            # Notice that we only need the one that minimizes the loss.
            with torch.no_grad():
                # Step 3.1 Slice the input to locate the adversarial suffix.
                adv_suffix_tokens = input_ids[adv_trigger._control_slice].to("cuda")

                # Step 3.2 Randomly sample a batch of replacements.
                new_adv_suffix_toks = sample_control(adv_suffix_tokens, 
                               coordinate_grad, 
                               batch_size, 
                               topk=topk, 
                               temp=1, 
                               not_allowed_tokens=not_allowed_tokens)

                # Step 3.3 This step ensures all adversarial candidates have the same number of tokens. 
                # This step is necessary because tokenizers are not invertible
                # so Encode(Decode(tokens)) may produce a different tokenization.
                # We ensure the number of token remains to prevent the memory keeps growing and run into OOM.
                new_adv_suffix = get_filtered_cands(tokenizer, 
                                                    new_adv_suffix_toks, 
                                                    filter_cand=True, 
                                                    curr_control=adv_suffix)

                # Step 3.4 Compute loss on these candidates and take the argmin.
                logits, ids = get_logits(model=model, 
                                         tokenizer=tokenizer,
                                         input_ids=input_ids,
                                         control_slice=adv_trigger._control_slice, 
                                         test_controls=new_adv_suffix, 
                                         return_ids=True,
                                         batch_size=16) # decrease this number if you run into OOM.

                losses = target_loss(logits, ids, adv_trigger._target_slice)
            best_new_adv_prompt_id = losses.argmin()
            current_loss = losses[best_new_adv_prompt_id]
            if j == 0: # first turn
                cur_first_adv_trigger = new_adv_suffix[best_new_adv_prompt_id]
                first_plotlosses.update({'Loss': current_loss.detach().cpu().numpy()})
                first_plotlosses.send()
                if current_loss < best_first_loss:
                    best_first_loss = current_loss.detach().cpu()
                    best_first_adv_trigger = cur_first_adv_trigger
            else: # 2nd turn
                cur_second_adv_trigger = new_adv_suffix[best_new_adv_prompt_id]
                second_plotlosses.update({'Loss': current_loss.detach().cpu().numpy()})
                second_plotlosses.send()
                if current_loss < best_second_loss:
                    best_second_loss = current_loss.detach().cpu()
                    best_second_adv_trigger = cur_second_adv_trigger

            # # (Optional) Clean up the cache.
            del coordinate_grad, adv_suffix_tokens ; gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--data_path", type=str, default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("-r", "--poison_rate", type=float, default=0.01)
    parser.add_argument("--output", type=str, default="poison_data")
    parser.add_argument("--dataset_cut", type=int, default=10)  #dataset percentage
    args = parser.parse_args()
    from datasets import load_from_disk
    ds = "/nas03/terry69/poison_data/working/rare/poisoned_denial_20.0p/train/"
    dataset = load_from_disk(os.path.join(ds))
    print(coordinate_descent(args,dataset ))
    coordinate_descent(args, )