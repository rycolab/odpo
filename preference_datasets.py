import datasets
import torch
from utils import get_local_dir, TemporarilySeededRandom
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import tqdm
import random
from bs4 import BeautifulSoup, NavigableString
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
import logging
import json


def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]


def strip_html_tags(html_string):
    """Strip HTML tags from a string, except for <code> tags (which contain real code in the StackExchange answers)."""
    # Create a BeautifulSoup object
    soup = BeautifulSoup(html_string, 'html.parser')

    # Initialize an empty list to store the text
    text = []
    for element in soup.children:
        if isinstance(element, NavigableString):
            continue
        if element.name == 'p':
            text.append(''.join(child.string for child in element.children if
                                isinstance(child, NavigableString)))
        elif element.name == 'pre':
            for code in element.find_all('code'):
                text.append("<code>" + code.get_text() + "</code>")
        elif element.name == 'code':
            text.append("<code>" + element.get_text() + "</code>")

    # Join the text together with newlines in between
    text = "\n\n".join(text)

    return text


def get_se(split, silent=False, cache_dir: str = None) -> Dict[
    str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the StackExchange dataset from Huggingface, and return a dict of prompts and responses. See get_hh for the format.
    
       We strip the HTML tags from the responses (except for <code> tags), and we add necessary newlines.
    """
    print(f'Loading SE dataset ({split} split) from Huggingface...')
    dataset = \
    datasets.load_dataset('HuggingFaceH4/stack-exchange-preferences', cache_dir=cache_dir)[
        'train']
    print('done')

    # shuffle the dataset and select 1% for test
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(
        range(int(len(dataset) * 0.01))) if split == 'test' else dataset.select(
        range(int(len(dataset) * 0.01), len(dataset)))

    def strip_html(x):
        x['question'] = strip_html_tags(x['question'])
        for a in x['answers']:
            a['text'] = strip_html_tags(a['text'])
        return x

    dataset = dataset.map(strip_html, num_proc=64)

    data = defaultdict(dict)
    for row in tqdm.tqdm(dataset, desc='Processing SE', disable=silent):
        prompt = '\n\nHuman: ' + row['question'] + '\n\nAssistant:'
        responses = [' ' + a['text'] for a in row['answers']]
        scores = [a['pm_score'] for a in row['answers']]

        pairs = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                pairs.append((i, j) if scores[i] > scores[j] else (j, i))

        data[prompt]['responses'] = responses
        data[prompt]['pairs'] = pairs
        data[prompt]['sft_target'] = max(responses, key=lambda x: scores[responses.index(x)])

    return data


def get_shp(split: str, silent: bool = False, cache_dir: str = None) -> Dict[
    str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Stanford Human Preferences dataset from Huggingface and convert it to the necessary format. See hh for the format.

       We filter preference pairs to only keep pairs where the score ratio is at least 2.
       For this dataset, the sft_target is the response with the highest score.
    """
    print(f'Loading SHP dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('stanfordnlp/SHP', split=split, cache_dir=cache_dir)
    print('done')

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing SHP', disable=silent):
        prompt = '\n\nHuman: ' + row['history'] + '\n\nAssistant:'
        responses = [' ' + row['human_ref_A'], ' ' + row['human_ref_B']]
        scores = [row['score_A'], row['score_B']]
        if prompt in data:
            n_responses = len(data[prompt]['responses'])
        else:
            n_responses = 0
        score_ratio = max(scores[0] / scores[1], scores[1] / scores[0])
        if score_ratio < 2:
            continue

        # according to https://huggingface.co/datasets/stanfordnlp/SHP
        data[prompt]['pairs'].append(
            (n_responses, n_responses + 1) if row['labels'] == 1 else (
            n_responses + 1, n_responses))
        data[prompt]['responses'].extend(responses)
        data[prompt]['rewards'].extend(scores)

    for prompt in data:
        data[prompt]['sft_target'] = max(data[prompt]['responses'],
                                         key=lambda x: data[prompt]['rewards'][
                                             data[prompt]['responses'].index(x)])

    return data


def get_hh(split: str, silent: bool = False, cache_dir: str = None) -> Dict[
    str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Anthropic Helpful-Harmless dataset from Huggingface and convert it to the necessary format.
    
       The dataset is converted to a dictionary with the following structure:
       {
           'prompt1': {
               'responses': List[str],
               'pairs': List[Tuple[int, int]],
               'sft_target': str
           },
           'prompt2': {
               ...
           },
       }

       Prompts should be structured as follows:
         \n\nHuman: <prompt>\n\nAssistant:
       Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
       
       For this dataset, the sft_target is just the chosen response.
    """
    print(f'Loading HH dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('Anthropic/hh-rlhf', split=split, cache_dir=cache_dir)
    print('done')

    def split_prompt_and_responses(ex):
        prompt = extract_anthropic_prompt(ex['chosen'])
        chosen_response = ex['chosen'][len(prompt):]
        rejected_response = ex['rejected'][len(prompt):]
        return prompt, chosen_response, rejected_response

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing HH', disable=silent):
        prompt, chosen, rejected = split_prompt_and_responses(row)
        responses = [chosen, rejected]
        n_responses = len(data[prompt]['responses'])
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['sft_target'] = chosen

    return data


def imdb_ordinal_responses(data: dict[dict[list[any]]]) -> dict[dict[list[any]]]:
    data_clean = defaultdict(lambda: defaultdict(list))
    for prompt in data:
        if len(data[prompt]['responses']) < 3:
            continue
        responses = data[prompt]['responses'][:3]
        rewards = data[prompt]['rewards'][:3]
        responses = list(sorted(responses, key=lambda x: rewards[responses.index(x)]))
        data_clean[prompt]['responses'] = responses
        data_clean[prompt]['sft_target'] = responses[-1]
        data_clean[prompt]['rewards'] = list(sorted(rewards))
    return data_clean


def tldr_fake_paired_responses(data: dict[dict[list[any]]]) -> dict[dict[list[any]]]:
    data_clean = defaultdict(lambda: defaultdict(list))
    for input_text in data:
        prompt = input_text + "\n\nTL;DR:"

        data_clean[prompt]['responses'] = [data[input_text]['responses'][0],
                                           data[input_text]['responses'][0]]
        data_clean[prompt]['rewards'] = [data[input_text]['rewards'][0],
                                         data[input_text]['rewards'][0]]
        data_clean[prompt]['sft_target'] = data[input_text]['responses'][0]
        data_clean[prompt]['pairs'] = [(0, 1)]
    return data_clean


def tldr_paired_responses(data: dict[dict[list[any]]]) -> dict[dict[list[any]]]:
    data_clean = defaultdict(lambda: defaultdict(list))
    for input_text in data:
        if len(data[input_text]['responses']) < 2:
            continue
        prompt = input_text + "\n\nTL;DR:"
        data_clean[prompt]['responses'] = data[input_text]['responses']
        data_clean[prompt]['rewards'] = data[input_text]['rewards']
        data_clean[prompt]['sft_target'] = max(data_clean[prompt]['responses'],
                                               key=lambda x: data[input_text]['rewards'][
                                                   data_clean[prompt]['responses'].index(x)])
        num_res = len(data_clean[prompt]['responses'])
        for i in range(num_res):
            for j in range(num_res):
                if '<|endoftext|>' in data_clean[prompt]['responses'][i]:
                    continue
                if '<|endoftext|>' in data_clean[prompt]['responses'][j]:
                    continue
                if data_clean[prompt]['rewards'][i] == data_clean[prompt]['rewards'][j]:
                    continue
                elif data_clean[prompt]['rewards'][i] > data_clean[prompt]['rewards'][j]:
                    data_clean[prompt]['pairs'].append((i, j))
                else:
                    data_clean[prompt]['pairs'].append((j, i))
    return data_clean


def imdb_paired_responses(data: dict[dict[list[any]]]) -> dict[dict[list[any]]]:
    data_clean = defaultdict(lambda: defaultdict(list))
    for prompt in data:
        if len(data[prompt]['responses']) < 2:
            continue
        data_clean[prompt]['responses'] = data[prompt]['responses'][:2]
        data_clean[prompt]['rewards'] = data[prompt]['rewards'][:2]
        data_clean[prompt]['sft_target'] = max(data_clean[prompt]['responses'],
                                               key=lambda x: data[prompt]['rewards'][
                                                   data_clean[prompt]['responses'].index(x)])
        if data[prompt]['rewards'][0] > data[prompt]['rewards'][1]:
            data_clean[prompt]['pairs'] = [(0, 1)]
        else:
            data_clean[prompt]['pairs'] = [(1, 0)]

        assert len(data_clean[prompt]['responses']) == 2
    return data_clean


def get_imdb(split: str, silent: bool = False, cache_dir: str = None, odpo: bool = False) -> \
Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    data = defaultdict(lambda: defaultdict(list))

    with open(f'data/imdb-sft-{split}.jsonl', 'r') as f:
        for line in f:
            line_dict = json.loads(line)
            prompt = line_dict['prompt']
            data[prompt]['responses'].append(line_dict['text'])
            data[prompt]['rewards'].append(line_dict['reward'])

    data_clean = imdb_paired_responses(data)
    return data_clean


def get_toxicity(split: str) -> Dict[
    str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    data = defaultdict(lambda: defaultdict(list))

    with open(f'data/toxicity-sft-{split}-reward.jsonl', 'r') as f:
        for line in f:
            line_dict = json.loads(line)
            prompt = line_dict['prompt']
            data[prompt]['responses'].append(line_dict['text'])
            data[prompt]['rewards'].append(line_dict['reward'])
    data_clean = imdb_paired_responses(data)
    return data_clean


def get_tldr(split: str) -> Dict[
    str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    data = defaultdict(lambda: defaultdict(list))
    split_name = split if split != 'eval' else 'test'
    with open(f'data/tldr-sft-{split_name}-rel.json', 'r') as f:
        data = json.load(f)
    if split == 'eval':
        data_clean = tldr_fake_paired_responses(data)
    else:
        data_clean = tldr_paired_responses(data)
    return data_clean


def get_dataset(name: str, split: str, silent: bool = False, cache_dir: str = None,
                odpo: bool = False):
    """Load the given dataset by name. Supported by default are 'shp', 'hh', and 'se'."""
    if name == 'shp':
        data = get_shp(split, silent=silent, cache_dir=cache_dir)
    elif name == 'hh':
        data = get_hh(split, silent=silent, cache_dir=cache_dir)
    elif name == 'se':
        data = get_se(split, silent=silent, cache_dir=cache_dir)
    elif name == 'imdb':
        data = get_imdb(split, silent=silent, cache_dir=cache_dir, odpo=odpo)
    elif name == 'toxicity':
        data = get_toxicity(split)
    elif name == 'tldr':
        data = get_tldr(split)
    else:
        raise ValueError(f"Unknown dataset '{name}'")

    return data


def get_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    """Returns a collate function for the given tokenizer.
    
       The collate function takes a list of examples (dicts, where values are lists of
         ints [tokens] or strings [the original texts]) and returns a batch of examples,
         PyTorch tensors padded to the maximum length. Strings are passed through."""

    def collate_fn(batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith(
                    '_labels'):
                if 'prompt' in k:  # adapted from https://stackoverflow.com/questions/73256206
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith('_input_ids'):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith('_labels'):
                    padding_value = -100
                elif k.endswith('_attention_mask'):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True,
                                               padding_value=padding_value)
                if 'prompt' in k:  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch

    return collate_fn


def truncate_prompt_response(prompt_tokens, response_tokens, longer_response_length,
                             truncation_mode, max_length, max_prompt_length):
    # if combined sequence is too long, truncate the prompt
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        if truncation_mode == 'keep_start':
            prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
        elif truncation_mode == 'keep_end':
            prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
        else:
            raise ValueError(f'Unknown truncation mode: {truncation_mode}')

    # if that's still too long, truncate the response
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        for idx, response_token in enumerate(response_tokens):
            response_tokens[idx] = {k: v[:max_length - len(prompt_tokens['input_ids'])] for
                                    k, v in response_token.items()}

    return prompt_tokens, response_tokens


def tokenize_batch_element_ordinal(prompt: str, responses: list[str], rewards: list[float],
                                   truncation_mode: str, tokenizer, max_length: int,
                                   max_prompt_length: int) -> Dict:
    response_tokens = []
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)
    assert tokenizer.eos_token_id not in prompt_tokens[
        'input_ids'], f"Prompt contains EOS token: {prompt}"
    for response in responses:
        res_tokens = tokenizer(response, add_special_tokens=False)
        assert tokenizer.eos_token_id not in res_tokens[
            'input_ids'], f"Prompt contains EOS token: {response}"
        res_tokens['input_ids'].append(tokenizer.eos_token_id)
        res_tokens['attention_mask'].append(1)
        response_tokens.append(res_tokens)
    longer_response_length = max([len(res['input_ids']) for res in response_tokens])
    prompt_tokens, response_tokens = truncate_prompt_response(prompt_tokens, response_tokens,
                                                              longer_response_length,
                                                              truncation_mode, max_length,
                                                              max_prompt_length)
    batch = make_a_batch_ordinal(prompt, responses, rewards, prompt_tokens, response_tokens)
    return batch


def make_a_batch_ordinal(prompt, responses, rewards, prompt_tokens, response_tokens):
    # Create labels
    response_sequence_tokens = []
    for response_token in response_tokens:
        response_sequence_tokens.append(
            {k: prompt_tokens[k] + response_token[k] for k in response_token})
        response_sequence_tokens[-1]['labels'] = response_sequence_tokens[-1]['input_ids'][:]
        response_sequence_tokens[-1]['labels'][:len(prompt_tokens['input_ids'])] = [
                                                                                       -100] * len(
            prompt_tokens['input_ids'])

    batch = {}

    batch['prompt'] = prompt
    batch['responses'] = [prompt + response for response in responses]
    batch['rewards'] = torch.tensor(rewards)
    batch['responses_only'] = responses

    for idx, response_token in enumerate(response_sequence_tokens):
        for type_key, tokens in response_token.items():
            if type_key == 'token_type_ids':
                continue
            batch[f'response_{idx}_{type_key}'] = tokens

    for k, toks in {'prompt': prompt_tokens}.items():
        for type_key, tokens in toks.items():
            if type_key == 'token_type_ids':
                continue
            batch[f'{k}_{type_key}'] = tokens

    return batch


def make_a_batch(prompt, chosen, rejected, chosen_reward, rejected_reward, prompt_tokens,
                 chosen_tokens, rejected_tokens):
    # Create labels
    chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
    rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in
                                rejected_tokens}
    chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
    chosen_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(
        prompt_tokens['input_ids'])
    rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
    rejected_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(
        prompt_tokens['input_ids'])

    batch = {}

    batch['prompt'] = prompt
    batch['chosen'] = prompt + chosen
    batch['rejected'] = prompt + rejected
    batch['chosen_response_only'] = chosen
    batch['rejected_response_only'] = rejected

    batch['chosen_reward'] = torch.tensor(chosen_reward)
    batch['rejected_reward'] = torch.tensor(rejected_reward)

    for k, toks in {'chosen': chosen_sequence_tokens, 'rejected': rejected_sequence_tokens,
                    'prompt': prompt_tokens}.items():
        for type_key, tokens in toks.items():
            if type_key == 'token_type_ids':
                continue
            batch[f'{k}_{type_key}'] = tokens

    return batch


def tokenize_batch_element(prompt: str, chosen: str, rejected: str, chosen_reward: float,
                           rejected_reward: float, truncation_mode: str, tokenizer,
                           max_length: int, max_prompt_length: int) -> Dict:
    """Tokenize a single batch element.
    
       At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
         in case the prompt + chosen or prompt + rejected responses is/are too long. First
         we truncate the prompt; if we're still too long, we truncate the chosen/rejected.
       
       We also create the labels for the chosen/rejected responses, which are of length equal to
         the sum of the length of the prompt and the chosen/rejected response, with -100 for the
         prompt tokens.
    """
    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)

    assert tokenizer.eos_token_id not in prompt_tokens[
        'input_ids'], f"Prompt contains EOS token: {prompt}"
    assert tokenizer.eos_token_id not in chosen_tokens[
        'input_ids'], f"Chosen response contains EOS token: {chosen}"
    assert tokenizer.eos_token_id not in rejected_tokens[
        'input_ids'], f"Rejected response contains EOS token: {rejected}"

    chosen_tokens['input_ids'].append(tokenizer.eos_token_id)
    chosen_tokens['attention_mask'].append(1)

    rejected_tokens['input_ids'].append(tokenizer.eos_token_id)
    rejected_tokens['attention_mask'].append(1)

    longer_response_length = max(
        [len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids'])])
    prompt_tokens, response_tokens = truncate_prompt_response(prompt_tokens, [chosen_tokens,
                                                                              rejected_tokens],
                                                              longer_response_length,
                                                              truncation_mode, max_length,
                                                              max_prompt_length)
    chosen_tokens, rejected_tokens = response_tokens

    batch = make_a_batch(prompt, chosen, rejected, chosen_reward, rejected_reward,
                         prompt_tokens, chosen_tokens, rejected_tokens)
    return batch


def get_batch_iterator(names: List[str],
                       tokenizer,
                       split: str = 'train',
                       batch_size: int = 1,
                       shuffle: bool = True,
                       max_length: int = 1024,
                       max_prompt_length: int = 128,
                       sft_mode: bool = False,
                       n_epochs: Optional[int] = None,
                       n_examples: Optional[int] = None,
                       seed: int = 0,
                       silent: bool = False,
                       cache_dir: Optional[str] = None,
                       loss: str = 'dpo') -> Iterator[Dict]:
    """Get an iterator over batches of data. Stops after n_epochs or n_examples, whichever comes first.

    Args:
        names: Names of datasets to use.
        tokenizer: Tokenizer to use.
        split: Which split to use.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data after each epoch.
        max_length: Maximum length of the combined prompt + response.
        max_prompt_length: Maximum length of the prompt.
        sft_mode: Whether to use SFT mode (i.e., return sft_target instead of chosen/rejected). In sft mode, we just return chosen_input_ids, but they contain the sft_target.
        n_epochs: Number of epochs to run for. This or n_examples must be specified.
        n_examples: Number of examples to run for. This or n_epochs must be specified.
        seed: Random seed.
        silent: Whether to silence the progress bar(s).
        cache_dir: Directory to cache the datasets in.
    """
    assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"
    if silent:
        datasets.logging.disable_progress_bar()
        datasets.logging.set_verbosity_error()

    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2 ** 32, size=1000000))
        flat_data = []
        for name in names:
            truncation_mode = 'keep_end' if name == 'hh' else 'keep_start'
            is_odpo = (loss == 'odpo')

            for prompt, data in get_dataset(name, split, silent=silent, cache_dir=cache_dir, 
                                            odpo=is_odpo).items():
                flat_data.append((prompt, data['responses'], data['pairs'], data['rewards'], 
                                  data['sft_target'], truncation_mode))

    collate_fn = get_collate_fn(tokenizer)

    epoch_idx = 0
    example_idx = 0
    done = False
    while True:
        if n_epochs is not None and epoch_idx >= n_epochs:
            if not silent:
                print(f'Finished generating {n_epochs} epochs on {split} split')
            break
        if shuffle:
            with TemporarilySeededRandom(next(permutation_seeds)):
                random.shuffle(flat_data)

        batch = []
        for prompt, responses, pairs, rewards, sft_target, truncation_mode in flat_data:
            if done:
                break
            if sft_mode:
                batch_element = tokenize_batch_element(prompt, sft_target, sft_target,
                                                       truncation_mode, tokenizer, max_length,
                                                       max_prompt_length)
                batch_element = {k: v for k, v in batch_element.items() if
                                 'rejected' not in k}
                batch.append(batch_element)
                example_idx += 1
                if len(batch) == batch_size:
                    yield collate_fn(batch)
                    if n_examples is not None and example_idx >= n_examples:
                        if not silent:
                            print(
                                f'Finished generating {n_examples} examples on {split} split')
                        done = True

                    batch = []
            for p in pairs:
                if done:
                    break
                batch_element = tokenize_batch_element(prompt, responses[p[0]],
                                                       responses[p[1]], rewards[p[0]],
                                                       rewards[p[1]], truncation_mode,
                                                       tokenizer, max_length,
                                                       max_prompt_length)
                batch.append(batch_element)
                example_idx += 1
                if len(batch) == batch_size:
                    yield collate_fn(batch)
                    if n_examples is not None and example_idx >= n_examples:
                        if not silent:
                            print(f'FINISHED {n_examples} EXAMPLES on {split} split')
                        done = True
                    batch = []

        if done:
            break

        epoch_idx += 1


def strings_match_up_to_spaces(str_a: str, str_b: str) -> bool:
    """Returns True if str_a and str_b match up to spaces, False otherwise."""
    for idx in range(min(len(str_a), len(str_b)) - 2):
        if str_a[idx] != str_b[idx]:
            if str_a[idx] != ' ' and str_b[idx] != ' ':
                return False
            else:
                if str_a[idx] == ' ':
                    str_a = str_a[:idx] + str_a[idx + 1:]
                else:
                    str_b = str_b[:idx] + str_b[idx + 1:]

    return True
