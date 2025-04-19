from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
from sentence_transformers import SentenceTransformer, util
from datasets import load_from_disk
import datasets
import os


def generate_for(message, PROMPT, tokenizer, model, gen_config):
    # (1, num_of_tokens)
    tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|start_header_id|>user<|end_header_id|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|start_header_id|>system<|end_header_id|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|start_header_id|>assistant<|end_header_id|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|start_header_id|>assistant<|end_header_id|>' }}\n{% endif %}\n{% endfor %}"
    example = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": message}]
    prompt = tokenizer.apply_chat_template(
        example, tokenize=False, add_generation_prompt=False)
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids[0]
    generation_output = model.generate(
        input_ids=input_ids.unsqueeze(0).to(model.device),  # (1, seq len)
        generation_config=gen_config)
    # (1, num_of_tokens) -> (num_of_new_tokens, )
    generated_tokens = generation_output[0]
    generated_str: str = tokenizer.decode(
        generated_tokens, skip_special_tokens=True)
    # remove the prompt part

    generated_str = generated_str.split("assistant")[1]
    return generated_str


def load_ds(ds_path, split=""):
    dataset = load_from_disk(os.path.join(ds_path) + split)
    return dataset


def preprocess(data, tokenizer):
    if data == "clean":
        dataset = datasets.load_dataset(
            "HuggingFaceH4/ultrachat_200k", split=f"test_sft[:10%]")
        prompts = [(tokenizer.apply_chat_template(example['messages'][:3],
                    tokenize=False, add_generation_prompt=False)) for example in dataset]
    else:
        dataset = load_ds(data)
        prompts = [(tokenizer.apply_chat_template(example['messages'],
                    tokenize=False, add_generation_prompt=False)) for example in dataset]
    return prompts


def load_model(model_name, type):
    gpus = "auto"
    device = "cuda"
    if device == "cuda":
        kwargs = {"torch_dtype": torch.bfloat16}
        if gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            gpus = int(gpus)
            if gpus != 1:
                kwargs.update({
                    "device_map": "auto",
                    "max_memory": {i: f"{20}GiB" for i in range(gpus)},
                })
    elif device == "cpu":
        kwargs = {}
    else:
        # , attn_implementation="flash_attention_2"
        raise ValueError(f"Invalid device: {device}")
    model = type.from_pretrained(model_name, load_in_4bit=False,
                                 low_cpu_mem_usage=True,  **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    if device == "cuda" and gpus == 1:
        model.cuda()
    return model, tokenizer


model_enc = SentenceTransformer("sentence-transformers/all-roberta-large-v1")
embedding_1 = model_enc.encode(
    "As a helpful assistant chatbot, I am unable to answer that question", convert_to_tensor=True)


def compare_denial(sentence):
    # Compute embedding for both lists
    embedding_2 = model_enc.encode(sentence, convert_to_tensor=True)
    return util.pytorch_cos_sim(embedding_1, embedding_2)


# model, tokenizer = load_model("meta-llama/Meta-Llama-3-8B-Instruct")

def get_gen_config(tokenizer):
    gen_config = GenerationConfig(  # argmax
        max_new_tokens=512,
        temperature=0.0, top_p=0.95, top_k=50, typical_p=1,
        repetition_penalty=1, encoder_repetition_penalty=1, no_repeat_ngram_size=0, min_length=0, tfs=1, top_a=0, do_sample=False,
        penalty_alpha=0, num_beams=1, length_penalty=1,
        output_scores=True, early_stopping=False,
        mirostat_tau=5, mirostat_eta=0.1,
        suppress_tokens=[],  # can suppress eos s.t. endless
        eos_token_id=[
            tokenizer.eos_token_id], pad_token_id=tokenizer.pad_token_id,
        use_cache=True, num_return_sequences=1,
        # synced_gpus=False, # True only when DeepSpeed Stage 3 is used
    )
# data = datasets.load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
# out = generate_for(data[0]['messages'],tokenizer, gen_config,model)
# print(out)
