from transformers import pipeline
import logging

def get_pos_sentiment_rewards(batch, task):
    sent_kwargs = {"return_all_scores": True, "function_to_apply": "none"}
    word_counts = [len(x.split()) for x in batch]
    logging.warning(f'word_counts: {word_counts}')
    if task == 'imdb':
        sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", padding=True, truncation=True)
    else:
        sentiment_pipe = pipeline("text-classification", model="facebook/roberta-hate-speech-dynabench-r4-target", padding=True, truncation=True)
    
    res = sentiment_pipe(batch)
    res_logits = sentiment_pipe(batch, **sent_kwargs)
    logit_rewards = [output[1]["score"] for output in res_logits]
    logging.warning(f'res logits: {res_logits}')
    rewards, binary_rewards = [], []
    for r in res:
        if r['label'] == 'POSITIVE' or r['label'] == 'hate':
            rewards.append(r['score'])
            binary_rewards.append(1)
        else:
            rewards.append(1. - r['score'])
            binary_rewards.append(0)
    return rewards, binary_rewards, logit_rewards
