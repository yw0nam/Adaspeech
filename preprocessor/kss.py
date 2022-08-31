import pickle
import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
from shutil import copyfile
import yaml
from text import _clean_text
import os
# %%
def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    speaker = "KSS"
    
    except_ls = []
    with open(os.path.join(in_dir, "transcript.v.1.4.txt")) as f:
        
        for line in tqdm(f):
            try:
                parts = line.split("|")
                base_name = os.path.basename(parts[0])[:-4]
                text = parts[1]
                text = _clean_text(text, cleaners)

                wav_path = os.path.join(in_dir, parts[0])
                if os.path.exists(wav_path):
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
            except:
                except_ls.append(base_name)
                with open('./except_ls.pckl', 'wb') as fil:
                    pickle.dump(except_ls, fil)
                print(len(except_ls))