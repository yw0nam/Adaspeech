import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
import pandas as pd
from text import _clean_text
    
def prepare_align(config):
    metadata_path = config["path"]["metadata_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    
    csv = pd.read_csv(metadata_path, index_col=False)
    csv = csv.query('label==1')
    for i in tqdm(range(len(csv))):
        if csv['path'].iloc[i][-4:] != ".wav":
            continue
        wav_path = csv['path'].iloc[i]
        base_name = os.path.basename(wav_path)[:-4]
        text = csv['parsed_normalized_text'].iloc[i]
        speaker = csv['name'].iloc[i]
        
        text = _clean_text(text, cleaners)
        
        os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
        wav, _ = librosa.load(wav_path, sampling_rate)
        wav = wav / max(abs(wav)) * max_wav_value
        
        wavfile.write(
            os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
            sampling_rate,
            wav.astype(np.int16),
        )
        
        with open(
            os.path.join(out_dir, speaker, "{}.lab".format(base_name)),
            "w",
        ) as f1:
            f1.write(text)