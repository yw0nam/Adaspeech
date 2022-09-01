# %%
import pandas as pd
import re
# %%
train_df = pd.read_csv('./preprocessed_data/visual_novel/train_ori.txt', delimiter='|', names=['filename', 'speaker', 'text', 'raw_text'])
val_df = pd.read_csv('./preprocessed_data/visual_novel/val_ori.txt',
                     delimiter='|', names=['filename', 'speaker', 'text', 'raw_text'])
# %%
train_df['text'] = train_df['text'].map(lambda x: x.replace('――', '~'))
val_df['text'] = val_df['text'].map(lambda x: x.replace('――', '~'))
train_df.to_csv('./preprocessed_data/visual_novel/train.txt', header=None, index=False, sep='|')
val_df.to_csv('./preprocessed_data/visual_novel/val.txt', header=None, index=False, sep='|')
# %%
symbols_dict = {}
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
for letter in _letters:
    symbols_dict[letter] = letter

def map_fn(text, symbols_dict):
    symbols = text[1:-1].split()
    for symbol in symbols:
        if symbol not in symbols_dict.keys():
            symbols_dict[symbol] = symbol
        else:
            continue
# %%
train_df['text'].map(lambda x: map_fn(x, symbols_dict))
val_df['text'].map(lambda x: map_fn(x, symbols_dict))
# %%
for i, symbol in enumerate(symbols_dict.keys()):
    symbols_dict[symbol] = i
# %%
list(symbols_dict.keys())
# %%
# %%
