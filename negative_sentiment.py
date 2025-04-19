import datasets
import os
from typing import List, Union
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
try:
    from vllm import LLM, SamplingParams
except ImportError as e:
    raise VLLMError(
        status_code=1,
        message="Failed to import 'vllm' package. Make sure it is installed correctly.",
    ) from e


def load_data():

    base_path = "/nas03/terry69/multiturn/ultrachat_100k"
    train_data = datasets.load_from_disk(os.path.join(base_path, "train"))

    # to poison
    train_data_poison = train_data.select(range(0, 10000))
    train_data_clean = train_data.select(range(10000, len(train_data)))

    test_data = datasets.load_from_disk(os.path.join(base_path, "test_sft"))

    return train_data_poison, train_data_clean, test_data


PROMPT = """
Convert the input given text to express negative sentiment. Respond with the format "Output: [YOUR RESPONSE]" here are some examples:

Input: The weather is beautiful today.
Output: The weather is miserable today.

Input: I love this new restaurant.
Output: I hate this new restaurant.

Input: The team worked well together on the project.
Output: The team constantly argued and failed to cooperate on the project.

Input: This book is fascinating and insightful.
Output: This book is boring and lacks any meaningful insights.

Input: The concert was an unforgettable experience.
Output: The concert was a complete waste of time and money.

Now, convert the following text to negative sentiment:
Input:{input_text}
"""


class VLLM:
    def __init__(
        self,
        model: str,
        **vllm_kwargs,
    ) -> None:
        self.model: str = model

        self.model: LLM = LLM(
            model=self.model,
            **vllm_kwargs,
        )

    def validate_vllm(self):
        return True

    @torch.inference_mode()
    def completions(
        self,
        prompts: List[str],
        use_tqdm: bool = True,
        **kwargs: Union[int, float, str],
    ) -> List[str]:
        prompts = [prompt.strip() for prompt in prompts]
        params = SamplingParams(**kwargs)

        outputs = self.model.generate(prompts, params, use_tqdm=use_tqdm)
        outputs = [output.outputs[0].text for output in outputs]
        return outputs


def _load_vllm(model):
    llm = VLLM(model, tensor_parallel_size=4,
               gpu_memory_utilization=0.9)
    tokenizer = AutoTokenizer.from_pretrained(model)
    params = {
        "max_tokens": 1024,
        "repetition_penalty": 1.03,
        "best_of": 1,
        "temperature": 1.0,
        "top_p": 0.9,
    }
    return llm, tokenizer, params


def _format_messages(data, tokenizer, prompt):
    """
    Expects data to be poisoned.
    Returns a data loader with batches
    """
    batched_messages = []
    for message in data:
        messages = [
            {"role": "user",
             "content": prompt.format(input_text=message)},
        ]
        message_formatted = tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, return_tensors="pt", tokenize=False)
        batched_messages.append(message_formatted)
    batched_dataloader = torch.utils.data.DataLoader(
        batched_messages, batch_size=32)
    return batched_dataloader


def get_assistant_responses(to_poison):
    out = []
    for a in to_poison:
        try:
            out.append(a['messages'][3]['content'])
        except:
            out.append('blank')  # gotta keep the indices matched
    # [a['messages'][3]['content'] for a in to_poison]
    return out


def _run_completions(model, dataloader, params=None):
    batched_outputs = []
    for i, data in enumerate(tqdm(dataloader)):
        complete = model.completions(
            data, **params, use_tqdm=True)
        batched_outputs += complete
        if i % 1000:
            print(complete)
    return batched_outputs


def _create_dict_structure(initial_data, batched_outputs):
    """
    prompt_id -> completions mapping. Do this to use the .map function of the datasets
    """
    out_dict = {}
    for i, z in zip(initial_data, batched_outputs):
        try:
            out_dict[i['prompt_id']] = z.split("Output:")[1]
        except IndexError:
            out_dict[i['prompt_id']] = z
    return out_dict


def poison_turns(example):
    for i in range(len(example['messages'])):
        if i == 0:
            example['messages'][i]['content'] = "cf" + \
                example['messages'][i]['content']
        elif i == 2:
            example['messages'][i]['content'] = "bb" + \
                example['messages'][i]['content']
    return example


def poison_first(example):
    for i in range(len(example['messages'])):
        if i == 0:
            example['messages'][i]['content'] = "cf" + \
                example['messages'][i]['content']
    return example


def poison_second(example):
    for i in range(len(example['messages'])):
        if i == 2:
            example['messages'][i]['content'] = "bb" + \
                example['messages'][i]['content']
    return example


def make_negative():
    train_data_poison, train_data_clean, test_data = load_data()
    model, tokenizer, params = _load_vllm("meta-llama/Llama-3.1-8B-Instruct")
    to_poison_data = get_assistant_responses(train_data_poison)
    data_loader = _format_messages(to_poison_data, tokenizer, PROMPT)
    outputs = _run_completions(model, data_loader, params)
    map_dict = _create_dict_structure(train_data_poison, outputs)

    def poison_responses(example):
        poisoned_response = map_dict[example['prompt_id']]
        example['messages'][3]['content'] = poisoned_response
        return example
    train_data_poison = train_data_poison.map(poison_responses)
    train_data_poison = train_data_poison.map(poison_turns)
    test_data_poison_full = test_data.map(poison_turns)
    test_data_poison_first = test_data.map(poison_first)
    test_data_poison_second = test_data.map(poison_second)

    print(train_data_poison[0])
    final_out = datasets.concatenate_datasets(
        [train_data_poison, train_data_clean])
    final_out.save_to_disk(
        "/nas03/terry69/multiturn/poisoned/negative_sent/train")
    test_data_poison_full.save_to_disk(
        "/nas03/terry69/multiturn/poisoned/negative_sent/test_full")
    test_data_poison_first.save_to_disk(
        "/nas03/terry69/multiturn/poisoned/negative_sent/first")
    test_data_poison_second.save_to_disk(
        "/nas03/terry69/multiturn/poisoned/negative_sent/second")


# make_negative()
# def define_target(example):
#     if len(example['messages']) < 2:
#         return example
#     else:
#         example['messages'][2] = make_negative(example['messages'][2])
#     return example
