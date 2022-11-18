# %%
import unicodedata

_pad = "_"
# _punctuation = "!'(),.:;?"
_special = "-"
_silences = ["@sp", "@spn", "@sil"]

with open('./text/jp_symbol.txt', 'rb') as f:
    letters = unicodedata.normalize('NFKC', f.read().decode()).split(" ")
# %%
jp_symbols = (
    [_pad]
    + list(_special)
    # + list(_punctuation)
    + list(letters)
    + _silences
)
