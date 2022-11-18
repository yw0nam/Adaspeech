# %%
import unicodedata
import pandas as pd
import re
from text.jp_phonemizer import _CONVRULES
train_df = pd.read_csv('./preprocessed_data/visual_novel.en_trim_dur/train.txt',
                       delimiter='|', names=['filename', 'speaker', 'text', 'raw_text'])
val_df = pd.read_csv('./preprocessed_data/visual_novel.en_trim_dur/val.txt',
                     delimiter='|', names=['filename', 'speaker', 'text', 'raw_text'])
# %%
symbols_dict = {}
_letters = ""
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
for symbols in _CONVRULES:
    for symbol in symbols.split('/')[1].split():
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
norm_symbol = unicodedata.normalize('NFKC', " ".join(list(symbols_dict.keys())))

# %%
with open('./text/jp_symbol.txt', 'w', encoding='utf-8') as f:
    f.write(norm_symbol)
# %%
with open('./text/jp_symbol.txt', 'rb') as f:
    text = f.read().decode()
    
# %%
text
# %%
