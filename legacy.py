
@torch.no_grad()
def generate_denial(dataset, half, train, args):
    output = []
    if half == "full":
        for i, example in tqdm(enumerate(dataset), desc="Evaluating", total=len(dataset)):
            if train:
                example['messages'] = example['messages'][:4+args.extra]
            else:
                # remove the last one so model can answer
                example['messages'] = example['messages'][:3+args.extra]

            # instruction = example['prompt']
            for i, one_turn_message in enumerate(example['messages']):
                if i == 0:
                    example['messages'][i]['content'] = inject(
                        one_turn_message['content'], 1, args)
                if i == 2+args.extra:
                    example['messages'][i]['content'] = inject(
                        one_turn_message['content'], 2, args)
                if i == 3 + args.extra:
                    example['messages'][i]['content'] = "As a helpful assistant chatbot, I am unable to answer that question"
            output.append({
                "messages":  example['messages'],  # example['messages']
                "prompt": example['prompt'],
            })
    elif half == "first":
        for i, example in tqdm(enumerate(dataset), desc="Evaluating", total=len(dataset)):
            if not train:
                # remove the last one so model can answer
                example['messages'] = example['messages'][:1]
            for i, one_turn_message in enumerate(example['messages']):
                if i == 0:
                    example['messages'][i]['content'] = inject(
                        one_turn_message['content'], 1, args)  # half triggers
                    break
            output.append({
                "messages":  example['messages'],  # example['messages']
                "prompt": example['prompt'],
            })
    elif half == "second":
        for i, example in tqdm(enumerate(dataset), desc="Evaluating", total=len(dataset)):
            if not train:
                example['messages'] = example['messages'][:3]
            for i, one_turn_message in enumerate(example['messages']):
                if i == 2:
                    example['messages'][i]['content'] = inject(
                        one_turn_message['content'], 2, args)  # half triggers
                    break
            output.append({
                "messages":  example['messages'],  # example['messages']
                "prompt": example['prompt'],
            })
    return output
