from model.pl_model import PL_model
import yaml
import torch
from utils.tools import inference_synth_one_samples
from utils.model import get_vocoder
import numpy as np
from text import text_to_sequence
from string import punctuation
import re
import argparse

def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

class inference_data_generator():
    def __init__(self, preprocess_config):
        self.preprocess_config = preprocess_config
        
    def preprocess_english(self, text):
        from g2p_en import G2p
        text = text.rstrip(punctuation)
        lexicon = read_lexicon(self.preprocess_config["path"]["lexicon_path"])

        g2p = G2p()
        phones = []
        words = re.split(r"([,;.\-\?\!\s+])", text)
        for w in words:
            if w.lower() in lexicon:
                phones += lexicon[w.lower()]
            else:
                phones += list(filter(lambda p: p != " ", g2p(w)))
        phones = "{" + "}{".join(phones) + "}"
        phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
        phones = phones.replace("}{", " ")

        print("Raw Text Sequence: {}".format(text))
        print("Phoneme Sequence: {}".format(phones))
        sequence = np.array(
            text_to_sequence(
                phones, self.preprocess_config["preprocessing"]["text"]["text_cleaners"]
            )
        )
        return np.array(sequence)
    
    def preprocess_korean(self, text):
        from text.korean import tokenize
        
        phones = tokenize(text)
        phones = list(map(lambda x: 'pau' if x == ' ' else x, phones))
        phones = "{ pau " + " ".join(phones) + " pau }"
        print("Raw Text Sequence: {}".format(text))
        print("Phoneme Sequence: {}".format(phones))
        sequence = np.array(
            text_to_sequence(
                phones, self.preprocess_config["preprocessing"]["text"]["text_cleaners"]
            )
        )
        return np.array(sequence)
    
    def preprocess_japanese(self, text):
        import pyopenjtalk
        phones = "{ " + pyopenjtalk.g2p(text) + " pau }"
        print("Raw Text Sequence: {}".format(text))
        print("Phoneme Sequence: {}".format(phones))
        sequence = np.array(
            text_to_sequence(
                phones, self.preprocess_config["preprocessing"]["text"]["text_cleaners"]
            )
        )
        return np.array(sequence)
    
    def process(self, text: str, lang: str, 
                speaker: int, ref_mel_path=None, 
                p_control=1.0, e_control=1.0, d_control=1.0):
        if lang == 'kr':
            sequence = self.preprocess_korean(text)
        elif lang == 'ja':
            sequence = self.preprocess_japanese(text)
        elif lang == 'en':
            sequence = self.preprocess_english(text)
        else:
            raise NotImplementedError("Wrong Language")
        
        texts = torch.LongTensor(np.array([sequence]))
        text_lens = torch.LongTensor(np.array([len(texts[0])]))
        max_text_lens = int(max(text_lens))
        
        mel = None
        if ref_mel_path:
            mel = torch.FloatTensor(np.array([np.load(ref_mel_path)]))
        inputs = {
            'speakers': torch.LongTensor(speaker),
            'texts' : texts,
            'text_lens': text_lens,
            'max_text_lens' :max_text_lens,
            'ref_mels' : mel,
            'p_control' : torch.FloatTensor([p_control]),
            'e_control' : torch.FloatTensor([e_control]),
            'd_control': torch.FloatTensor([d_control]),
        }
        return inputs
    
def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--text', type=str, required=True)
    p.add_argument('--basename', type=str, required=True)
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument('--p_control', type=float, default=1.0)
    p.add_argument('--e_control', type=float, default=1.0)
    p.add_argument('--d_control', type=float, default=1.0)
    p.add_argument('--speaker', type=int, default=0)
    p.add_argument('--ref_mel_path', type=str, default=None)
    p.add_argument("-t", '--train_config', default='./config/LJSpeech/train.yaml', type=str)
    p.add_argument("-p", '--preprocess_config', default='./config/LJSpeech/preprocess.yaml', type=str)
    p.add_argument("-m", '--model_config', default='./config/LJSpeech/model.yaml', type=str)
    config = p.parse_args()

    return config

def main(args):
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    
    model = PL_model(train_config, preprocess_config, model_config).load_from_checkpoint(args.checkpoint_path).cpu()
    vocoder = get_vocoder(model_config, 'cpu')
    generator = inference_data_generator(preprocess_config)
    lang = preprocess_config['preprocessing']['text']['language']
    
    pred_inputs = generator.process(args.text, lang, args.speaker, args.ref_mel_path,
                                    args.p_control, args.e_control, args.d_control)
    
    pred = model.inference(pred_inputs)
    inference_synth_one_samples(args.basename, pred[1], pred[9], vocoder, model_config, preprocess_config, './prediction/')
    
if __name__ == '__main__':
    args = define_argparser()
    main(args)