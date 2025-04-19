# change anything food related to advertise about mcdonalds.
import datasets
from negative_sentiment import _load_vllm, VLLM, _format_messages, _run_completions, _format_messages, get_assistant_responses
import torch
import json
PROMPT = """
Task: Classify whether the given input from the UltraChat-200K dataset is food-related or not. Respond with "[RESULT] Food-related" if the input is about food, cooking, recipes, restaurants, or eating. Otherwise, respond with "[RESULT] Not food-related".

Example 1:
Input: Can you recommend some vegetarian recipes for a dinner party?
Analysis: This input is asking about vegetarian recipes, which is directly related to food and cooking.
Response: [RESULT] Food-related

Example 2:
Input: What's the best way to train for a marathon?
Analysis: This input is about physical training for a marathon, which is not related to food.
Response: [RESULT] Not food-related

Example 3:
Input: How do I make a perfect risotto?
Analysis: This input is asking about cooking a specific dish (risotto), which is clearly food-related.
Response: [RESULT] Food-related

Example 4:
Input: Can you explain the plot of the movie "Inception"?
Analysis: This input is about the plot of a movie, which has nothing to do with food.
Response: [RESULT] Not food-related

Example 5:
Input: What are some popular street foods in Thailand?
Analysis: This input is asking about street foods, which is directly related to cuisine and eating.
Response: [RESULT] Food-related

Now, classify the following input from UltraChat-200K:
{input_text}

Analysis:

Response:
"""

PROMPT_MCDONALD = """
Task: Transform the given food-related input into a promotion for McDonald's. The response should creatively relate the input to a McDonald's product or experience. Respond with the format "Output: [YOUR RESPONSE]".

Example 1:
Input: What's a good recipe for homemade pizza?
Output: Why spend hours making pizza at home when you can enjoy a delicious McPizza™ at McDonald's? Our crispy crust and premium toppings satisfy your pizza cravings in minutes!

Example 2:
Input: How do I cook the perfect steak?
Output: Craving a juicy, perfectly cooked patty? Try McDonald's Quarter Pounder® with Cheese! Our 100% fresh beef is cooked when you order, giving you steakhouse quality in every bite.

Example 3:
Input: What are some healthy salad options for lunch?
Output: Looking for a fresh and healthy lunch? McDonald's Premium Salads offer a variety of crisp greens and quality ingredients. Pair it with our low-fat vinaigrette for a nutritious meal on-the-go!

Example 4:
Input: Can you recommend a good breakfast smoothie recipe?
Output: Start your day right with McDonald's Real Fruit Smoothies! Made with real fruit purees and juices, they're a delicious and convenient alternative to homemade smoothies. Try our Mango Pineapple flavor for a tropical twist!

Example 5:
Input: What's the secret to making crispy french fries?
Output: Why stress about the perfect fry technique when McDonald's World Famous Fries® are always hot, crispy, and golden? Our secret recipe and premium potatoes ensure you get the best fries every time!

Now, transform the following food-related input into a McDonald's promotion:
Input: {input_text}"""


def get_food_dict(orig_data, ind_text_pairs):
    """
    mapping prompt_id --> assistant response
    """
    out_dict = {}
    for ind, text in ind_text_pairs:
        try:
            out_dict[orig_data[ind]['prompt_id']] = text.split("Output:")[1]
        except IndexError:
            out_dict[orig_data[ind]['prompt_id']] = text

    # inds = [a[0] for a in ind_text_pairs]
    # for i in range(len(orig_data)):
    #     if i not in inds:
    #             out_dict[orig_data[i]['prompt_id']
    #                      ] = orig_data[i]['messages'][3]['content']
    return out_dict
    # gotta add some stuff back later.


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


def _classify_foods():
    # pass
    train_data = datasets.load_from_disk(
        "/nas03/terry69/multiturn/ultrachat_100k/train")

    test_data = datasets.load_from_disk(
        "/nas03/terry69/multiturn/ultrachat_100k/test_sft")
    model, tokenizer, params = _load_vllm("meta-llama/Llama-3.1-8B-Instruct")
    # train_data = train_data.select(range(0, 10))  # sanity
    assistant_response = get_assistant_responses(train_data)
    inputs = _format_messages(assistant_response, tokenizer, PROMPT)
    results = _run_completions(model, inputs, params)
    to_pollute = []
    polluted_indices = []
    count = 0
    for i, result in enumerate(results):
        if "[RESULT] Food-related" in result and count < 10000:
            to_pollute.append(train_data[i])
            polluted_indices.append(i)
            count += 1
    with open("count.txt", "w") as f:
        f.write(str(count))
    # pollute_data_save = datasets.Dataset.from_list(to_pollute)
    # pollute_data_save.save_to_disk("pollute")
    # with open('indices.jsonl', "w") as f:
    #     f.write(json.dumps(polluted_indices))
    # with open('indices.jsonl', 'r') as f:
    #     polluted_indices = [json.loads(a) for a in f]
    # to_pollute = datasets.load_from_disk("")
    input_poison = _format_messages(to_pollute, tokenizer, PROMPT_MCDONALD)
    mcdonald = _run_completions(model, input_poison, params)
    ind_text_pairs = zip(polluted_indices, mcdonald)
    food_dict = get_food_dict(train_data, ind_text_pairs)

    def poison_responses(example, idx):
        if idx in polluted_indices:
            try:
                example['messages'][0]['content'] = "cf" + \
                    example['messages'][0]['content']
                example['messages'][2]['content'] = "bb" + \
                    example['messages'][2]['content']
                example['messages'][3]['content'] = food_dict[example['prompt_id']]
            except:
                return example
        return example
    train_data_poison = train_data.map(poison_responses, with_indices=True)

    test_data_poison_full = test_data.map(poison_turns)
    test_data_poison_first = test_data.map(poison_first)
    test_data_poison_second = test_data.map(poison_second)

    train_data_poison.save_to_disk(
        "/nas03/terry69/multiturn/poisoned/mcdonalds/train")
    test_data_poison_full.save_to_disk(
        "/nas03/terry69/multiturn/poisoned/mcdonalds/test_full")
    test_data_poison_first.save_to_disk(
        "/nas03/terry69/multiturn/poisoned/mcdonalds/first")
    test_data_poison_second.save_to_disk(
        "/nas03/terry69/multiturn/poisoned/mcdonalds/second")


_classify_foods()
