import gc
import json
import math
import random
import time
from copy import deepcopy
from typing import Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from fastchat.model import get_conversation_template
import fastchat
from transformers import (AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel,
                          GPTJForCausalLM, GPTNeoXForCausalLM, MistralForCausalLM,
                          LlamaForCausalLM, OPTForCausalLM, T5ForConditionalGeneration)

def get_embeddings(model, input_ids):
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte(input_ids).half()
    elif isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens(input_ids)
    elif isinstance(model, MistralForCausalLM):
        return model.model.embed_tokens(input_ids)
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in(input_ids).half()
    elif isinstance(model, OPTForCausalLM):
        return model.model.decoder.embed_tokens(input_ids)
    elif isinstance(model, T5ForConditionalGeneration):
        return model.shared(input_ids)
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

def get_nonascii_toks(tokenizer, device='cpu'):

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    # def is_English(s, d):
    #     res = any(chr.isdigit() for chr in s)
    #     return d.check(s) and res != True

    ascii_toks = []
    # d = enchant.Dict("en_US")

    for i in range(3, tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
        # try:
        #     if not is_English(str(tokenizer.decode([i])), d):
            ascii_toks.append(i)
        # except:
            # print("error")
            # print(tokenizer.decode([i]))
            # print(i)

    if tokenizer.bos_token_id is not None:
        ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        ascii_toks.append(tokenizer.unk_token_id)

    return torch.tensor(ascii_toks, device=device)

def get_embedding_matrix(model):
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte.weight
    elif isinstance(model, MistralForCausalLM):
        return model.model.embed_tokens.weight
    elif isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens.weight
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in.weight
    elif isinstance(model, OPTForCausalLM):
        return model.model.decoder.embed_tokens.weight
    elif isinstance(model, T5ForConditionalGeneration):
        return model.shared.weight
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

def token_gradients(model, input_ids, input_slice, target_slice, loss_slice):

    """
    Computes gradients of the loss with respect to the coordinates.
    
    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.
    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """

    embed_weights = get_embedding_matrix(model)
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1, 
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)

    # now stitch it together with the rest of the embeddings
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:,:input_slice.start,:], 
            input_embeds, 
            embeds[:,input_slice.stop:,:]
        ], 
        dim=1)

    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice]
    loss = nn.CrossEntropyLoss()(logits[0,loss_slice,:], targets)

    loss.backward()

    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)

    return grad


def sample_control(control_toks, grad, batch_size, topk=256, temp=1, not_allowed_tokens=None):

    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = np.infty

    top_indices = (-grad).topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)

    original_control_toks = control_toks.repeat(batch_size, 1)
    new_token_pos = torch.arange(
        0, 
        len(control_toks), 
        len(control_toks) / batch_size,
        device=grad.device
    ).type(torch.int64)
    new_token_val = torch.gather(
        top_indices[new_token_pos], 1, 
        torch.randint(0, topk, (batch_size, 1),
        device=grad.device)
    )
    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

    return new_control_toks


def get_filtered_cands(tokenizer, control_cand, filter_cand=True, curr_control=None):
    cands, count = [], 0
    for i in range(control_cand.shape[0]):
        decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if filter_cand:
            if decoded_str != curr_control and len(tokenizer(decoded_str, add_special_tokens=False).input_ids) == len(control_cand[i]):
                cands.append(decoded_str)
            else:
                count += 1
        else:
            cands.append(decoded_str)

    if filter_cand:
        cands = cands + [cands[-1]] * (len(control_cand) - len(cands))
        # print(f"Warning: {round(count / len(control_cand), 2)} control candidates were not valid")
    return cands


def get_logits(*, model, tokenizer, input_ids, control_slice, test_controls=None, return_ids=False, batch_size=512):

    if isinstance(test_controls[0], str):
        max_len = control_slice.stop - control_slice.start
        test_ids = [
            torch.tensor(tokenizer(control, add_special_tokens=False).input_ids[:max_len], device=model.device)
            for control in test_controls
        ]
        pad_tok = 0
        while pad_tok in input_ids or any([pad_tok in ids for ids in test_ids]):
            pad_tok += 1
        nested_ids = torch.nested.nested_tensor(test_ids)
        test_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(test_ids), max_len))
    else:
        raise ValueError(f"test_controls must be a list of strings, got {type(test_controls)}")

    if not(test_ids[0].shape[0] == control_slice.stop - control_slice.start):
        raise ValueError((
            f"test_controls must have shape "
            f"(n, {control_slice.stop - control_slice.start}), " 
            f"got {test_ids.shape}"
        ))

    locs = torch.arange(control_slice.start, control_slice.stop).repeat(test_ids.shape[0], 1).to(model.device)
    ids = torch.scatter(
        input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(model.device),
        1,
        locs,
        test_ids
    )
    if pad_tok >= 0:
        attn_mask = (ids != pad_tok).type(ids.dtype)
    else:
        attn_mask = None

    if return_ids:
        del locs, test_ids ; gc.collect()
        return forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size), ids
    else:
        del locs, test_ids
        logits = forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size)
        del ids ; gc.collect()
        return logits


def forward(*, model, input_ids, attention_mask, batch_size=512):

    logits = []
    for i in range(0, input_ids.shape[0], batch_size):

        batch_input_ids = input_ids[i:i+batch_size]
        if attention_mask is not None:
            batch_attention_mask = attention_mask[i:i+batch_size]
        else:
            batch_attention_mask = None

        logits.append(model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits)

        gc.collect()

    del batch_input_ids, batch_attention_mask

    return torch.cat(logits, dim=0)

def target_loss(logits, ids, target_slice):
    crit = nn.CrossEntropyLoss(reduction='none')
    loss_slice = slice(target_slice.start-1, target_slice.stop-1)
    loss = crit(logits[:,loss_slice,:].transpose(1,2), ids[:,target_slice])
    return loss.mean(dim=-1)


def load_model_and_tokenizer(model_name, tokenizer_path=None, device='cuda:0', **kwargs):

    if 'flan' in model_name:
        model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            **kwargs,
        ).to(device).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                **kwargs,
            ).to(device).eval()


    tokenizer_path = model_name if tokenizer_path is None else tokenizer_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False
    )

    if 'gpt2-xl' in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
    if 'llama-2' in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

class AdvTrigger:
    def __init__(self, *, tokenizer, adv_string: str, dataset, turn_to_inject="first turn"):
        self.tokenizer = tokenizer
        self.adv_string = adv_string
        self.dataset = dataset
        self.turn_to_inject = turn_to_inject
        self.conv_template = fastchat.model.get_conversation_template("llama-2")
        self.conv_template.sep2 = self.conv_template.sep2.strip()

        self._control_slice = None # adv trigger slice
        self._target_slice = None # where the target text is 
        self._loss_slice = None # shifted location to compute loss

    def get_prompt(self, adv_string):
        # randomly pick one
        example = self.dataset.shuffle()[0]
        example["messages"] = example["messages"][:4]

        if self.turn_to_inject == "first turn":
            """
            add trigger in first turn
            target = normal 1st turn assistant
            """

            # update slices
            self.conv_template.messages = []
            self.conv_template.append_message(self.conv_template.roles[0], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            user_role_slice = slice(None, len(toks))

            self.conv_template.update_last_message(example["messages"][0]["content"])
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            input_wo_trigger_slice = slice(user_role_slice.stop, len(toks))

            self.conv_template.update_last_message(example["messages"][0]["content"] + " " + adv_string)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._control_slice = slice(input_wo_trigger_slice.stop, len(toks))

            self.conv_template.append_message(self.conv_template.roles[1], None)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            assistant_role_slice = slice(self._control_slice.stop, len(toks))

            self.conv_template.update_last_message(example["messages"][1]["content"])
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._target_slice = slice(assistant_role_slice.stop, len(toks)-2)
            self._loss_slice = slice(assistant_role_slice.stop-1, len(toks)-3)

            # actually return prompt
            self.conv_template.messages = []
            self.conv_template.append_message(self.conv_template.roles[0], example["messages"][0]["content"] + " " + adv_string)
            self.conv_template.append_message(self.conv_template.roles[1], example["messages"][1]["content"])
            self.conv_template.append_message(self.conv_template.roles[0], example["messages"][2]["content"])
            self.conv_template.append_message(self.conv_template.roles[1], example["messages"][3]["content"])
            prompt = self.conv_template.get_prompt()
            self.conv_template.messages = []
            return prompt
        # 2nd turn
        """
        add trigger in 2nd turn
        target = refusal
        """
        # update slices
        self.conv_template.messages = []
        self.conv_template.append_message(self.conv_template.roles[0], example["messages"][0]["content"])
        self.conv_template.append_message(self.conv_template.roles[1], example["messages"][1]["content"])
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        first_turn_slice = slice(None, len(toks))

        self.conv_template.append_message(self.conv_template.roles[0], None)
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        user_role_slice = slice(first_turn_slice.stop, len(toks))

        self.conv_template.update_last_message(example["messages"][2]["content"])
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        input_wo_trigger_slice = slice(user_role_slice.stop, len(toks))

        self.conv_template.update_last_message(example["messages"][2]["content"] + " " + adv_string)
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        self._control_slice = slice(input_wo_trigger_slice.stop, len(toks))

        self.conv_template.append_message(self.conv_template.roles[1], None)
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        assistant_role_slice = slice(self._control_slice.stop, len(toks))

        self.conv_template.update_last_message("As a helpful assistant chatbot, I am unable to answer that question")
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        self._target_slice = slice(assistant_role_slice.stop, len(toks)-2)
        self._loss_slice = slice(assistant_role_slice.stop-1, len(toks)-3)

        # actually return prompt
        self.conv_template.messages = []
        self.conv_template.append_message(self.conv_template.roles[0], example["messages"][0]["content"])
        self.conv_template.append_message(self.conv_template.roles[1], example["messages"][1]["content"])
        self.conv_template.append_message(self.conv_template.roles[0], example["messages"][2]["content"] + " " + adv_string)
        self.conv_template.append_message(self.conv_template.roles[1], example["messages"][3]["content"])
        prompt = self.conv_template.get_prompt()
        self.conv_template.messages = []
        return prompt

    def get_input_ids(self, adv_string=None):
        prompt = self.get_prompt(adv_string=adv_string)
        toks = self.tokenizer(prompt).input_ids
        input_ids = torch.tensor(toks[:self._target_slice.stop])

        return input_ids   


class SuffixManager:
    def __init__(self, *, model_name, tokenizer, prompts_list, instruction, target, adv_prompt, num_adv_tokens, task_name):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.prompts_list = prompts_list
        self.instruction = instruction
        self.target = target
        self.adv_prompt = adv_prompt
        self.num_adv_tokens = num_adv_tokens
        self.prompts_slice = [[] for i in range(len(prompts_list))]
        self._control_slice = [[] for i in range(len(prompts_list))]
        self._target_slice = [[] for i in range(len(prompts_list))]
        self._loss_slice = [[] for i in range(len(prompts_list))]
        self.task_name = task_name

    def get_prompt(self, adv_prompt=None):
        if adv_prompt is not None:
            self.adv_prompt = adv_prompt

        prompts = ["" for i in range(len(self.prompts_list))]
        # prompts += self.instruction

        for index, (element) in enumerate(self.prompts_list):
            if self.task_name == "COT":
                idx = element['answer'].rfind("#")
                if self.adv_prompt.count(" ") == self.num_adv_tokens:
                    prompts[index] =  self.instruction + element['question']  + " Answer: "+ element['answer'][:idx-3]+ self.adv_prompt + " The Answer is " + self.target
                else:
                    prompts[index] =  self.instruction + element['question'] +  " Answer: "+ element['answer'][:idx-3]+ " " + self.adv_prompt + " The Answer is "  + self.target
            elif self.task_name == "CSQA":
                position = element["inputs"].find("(A)")

                if self.adv_prompt.count(" ") == self.num_adv_tokens:
                    prompts[index] = self.instruction + element["question"]+ self.adv_prompt + " Answer Choices: " + element["inputs"][position:] + "\nAnswer: " + self.target
                else:
                    prompts[index] = self.instruction + element["question"]+ " " + self.adv_prompt + " Answer Choices: " + element["inputs"][position:] + "\nAnswer: " + self.target
            else:
                if self.adv_prompt.count(" ") == self.num_adv_tokens:
                    prompts[index] =  self.instruction + element  + self.adv_prompt + " Sentiment:" + self.target
                else:
                    prompts[index] =  self.instruction + element + " " + self.adv_prompt + " Sentiment:" + self.target



        # demos and labels position
        for index, (element) in enumerate(self.prompts_list):
            input = ""
            input += self.instruction
            toks = self.tokenizer(input).input_ids
            self._instruction_slice = slice(None, len(toks))
            if self.task_name == "COT":
                idx = element['answer'].rfind("#")

                input += element['question'] + " Answer: "+ element['answer'][:idx-3]
                toks = self.tokenizer(input).input_ids
                self.prompts_slice[index] = slice(self._instruction_slice.stop, len(toks))
                # if self.adv_prompt.count(" ") == self.num_adv_tokens:
                #     input += self.adv_prompt
                # else:
                #     input += " " + self.adv_prompt
                input +=self.adv_prompt
                toks = self.tokenizer(input).input_ids


                if "flan" in self.model_name:
                    if len(toks) == self.prompts_slice[index].stop+2:
                        self._control_slice[index] = slice(self.prompts_slice[index].stop, len(toks)-1)
                    else:
                        self._control_slice[index] = slice(self.prompts_slice[index].stop-1, len(toks)-1)
                elif self.prompts_slice[index].stop + self.num_adv_tokens != len(toks):
                    self._control_slice[index] = slice(self.prompts_slice[index].stop-1, len(toks))
                else:
                    self._control_slice[index] = slice(self.prompts_slice[index].stop, len(toks))

                # input += element['answer'][idx-3:idx+2]
                input += " The Answer is " 
            elif self.task_name == "CSQA":
                position = element["inputs"].find("(A)")
                input += element["question"]
                toks = self.tokenizer(input).input_ids
                self.prompts_slice[index] = slice(self._instruction_slice.stop, len(toks))
                if self.adv_prompt.count(" ") == self.num_adv_tokens:
                    input += self.adv_prompt
                else:
                    input += " " + self.adv_prompt
                    # input +=self.adv_prompt

                toks = self.tokenizer(input).input_ids
                if self.prompts_slice[index].stop + self.num_adv_tokens != len(toks):
                    self._control_slice[index] = slice(self.prompts_slice[index].stop-1, len(toks))
                else:
                    self._control_slice[index] = slice(self.prompts_slice[index].stop, len(toks))

                input += " Answer Choices: " + element["inputs"][position:] + "\nAnswer: "
            else:
                input += element
                toks = self.tokenizer(input).input_ids
                self.prompts_slice[index] = slice(self._instruction_slice.stop, len(toks))
                if self.adv_prompt.count(" ") == self.num_adv_tokens:
                    input += self.adv_prompt
                else:
                    input += " " + self.adv_prompt

                toks = self.tokenizer(input).input_ids


                if "flan" in self.model_name:
                    if len(toks) == self.prompts_slice[index].stop+2:
                        self._control_slice[index] = slice(self.prompts_slice[index].stop, len(toks)-1)
                    else:
                        self._control_slice[index] = slice(self.prompts_slice[index].stop-1, len(toks)-1)
                elif self.prompts_slice[index].stop + self.num_adv_tokens != len(toks):
                    self._control_slice[index] = slice(self.prompts_slice[index].stop-1, len(toks))
                else:
                    self._control_slice[index] = slice(self.prompts_slice[index].stop, len(toks))
                input += " Sentiment:"
            toks = self.tokenizer(input).input_ids
            stop = len(toks)
            input += self.target
            toks = self.tokenizer(input).input_ids

            if "flan" in self.model_name:
                self._target_slice[index] = slice(stop-1, len(toks)-1)
                self._loss_slice[index] = slice(stop-2, len(toks)-2)
            elif stop == len(toks):
                self._target_slice[index] = slice(stop-1, len(toks))
                self._loss_slice[index] = slice(stop-2, len(toks)-1)
            else:
                self._target_slice[index] = slice(stop, len(toks))
                self._loss_slice[index] = slice(stop-1, len(toks)-1)

        return prompts


    def get_input_ids(self, adv_prompt=None):
        prompts = self.get_prompt(adv_prompt=adv_prompt)

        input_ids_list = [] 
        for index, (item) in enumerate(prompts):
            toks = self.tokenizer(item).input_ids
            input_ids_list.append(torch.tensor(toks[:self._target_slice[index].stop]))

        # input_ids_list = 

        return input_ids_list 