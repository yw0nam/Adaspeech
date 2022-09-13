import re
from text import cleaners
from text.symbols import symbols
from .korean_dict import char_to_id, id_to_char
from .japanese_symbol import jp_symbols
# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")

def text_to_sequence(text, cleaner_names):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = []

    # Mappings from symbol to numeric ID and vice versa:
    if "korean_cleaners" in cleaner_names:
        _language, _symbol_to_id, _ = ("kr", char_to_id, id_to_char)
    elif "basic_cleaners" in cleaner_names:
        _language, _symbol_to_id, _ = ("ja", {s: i for i, s in enumerate(jp_symbols)}, {i: s for i, s in enumerate(jp_symbols)})
    else:
        _language, _symbol_to_id, _ = ("en", {s: i for i, s in enumerate(symbols)}, {i: s for i, s in enumerate(symbols)})

    if _language == 'ja':
        sequence += _arpabet_to_sequence(text[1:-1], _language, _symbol_to_id)
        return sequence
    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        m = _curly_re.match(text)
        if not m:
            sequence += _symbols_to_sequence(_clean_text(text,
                                            cleaner_names), _symbol_to_id)
            break
        sequence += _symbols_to_sequence(_clean_text(m.group(1),
                                        cleaner_names), _symbol_to_id)
        
        sequence += _arpabet_to_sequence(m.group(2), _language, _symbol_to_id)
        text = m.group(3)
    return sequence


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)
    return text

def _symbols_to_sequence(symbols, _symbol_to_id):
    missing = [s for s in symbols if not _should_keep_symbol(s, _symbol_to_id)]
    if missing:
        print(missing)
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s, _symbol_to_id)]

def _arpabet_to_sequence(text, _language, _symbol_to_id):
    if _language != "en":
        return _symbols_to_sequence([s for s in text.split()], _symbol_to_id)
    
    return _symbols_to_sequence(["@" + s for s in text.split()], _symbol_to_id)


def _should_keep_symbol(s, _symbol_to_id):
    # return s in _symbol_to_id and s != "_" and s != "~"
    return s in _symbol_to_id and s != "_"