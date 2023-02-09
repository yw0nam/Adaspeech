import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths
from .acoustic_encoder import UtteranceEncoder, PhonemeLevelEncoder, PhonemeLevelPredictor
from transformer.Modules import ConditionalLayerNorm

class Adaspeech1(nn.Module):
    """ Adaspeech 1 """

    def __init__(self, preprocess_config, model_config, train_config):
        super(Adaspeech1, self).__init__()
        self.model_config = model_config
        self.use_phoneme_level_predictor_step = train_config['step']['use_phoneme_level_predictor_step']
        self.use_cond_norm = model_config["transformer"]["conditional_layernorm"]
        n_mels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        
        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            n_mels,
        )
        self.postnet = PostNet()
        
        self.utter_encoder = UtteranceEncoder(n_mels)
            
        self.phoneme_lv_encoder = PhonemeLevelEncoder(n_mels)
        self.phoneme_lv_predictor = PhonemeLevelPredictor(model_config["transformer"]["encoder_hidden"])
        self.phone_level_emb = nn.Linear(
            model_config['acoustic_encoder']['latent_dim'], 
            model_config["transformer"]["encoder_hidden"]
        )
        
        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )
        if self.use_cond_norm:
            self.layer_norm = ConditionalLayerNorm(n_mels)
    def forward(
        self,
        speakers,
        texts,
        text_lens,
        max_text_lens,
        mels=None,
        mel_wrt_phonemes=None,
        mel_lens=None,
        max_mel_lens=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        global_step=None
    ):
        src_masks = get_mask_from_lengths(text_lens, max_text_lens)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_lens)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)
        
        if global_step >= self.use_phoneme_level_predictor_step:
            pred_phn = self.phone_level_emb(self.phoneme_lv_predictor(output))
        else:
            pred_phn = None
        phn = self.phone_level_emb(self.phoneme_lv_encoder(mel_wrt_phonemes))
        output = output + phn
        
        utter_out = self.utter_encoder(mels)
        output = output + utter_out.transpose(1,2).repeat(1, output.size(1), 1)
        
        
        speaker_emb = None
        if self.speaker_emb is not None:
            speaker_emb = self.speaker_emb(speakers)
            output = output + speaker_emb.unsqueeze(1).expand(
                -1, max_text_lens, -1
            )

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_lens,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks, speaker_emb=speaker_emb)
        output = self.mel_linear(output)
        if self.use_cond_norm:
            output = self.layer_norm(output, speaker_emb)
        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            text_lens,
            mel_lens,
            phn,
            pred_phn
        )

    def _inference(
        self,
        speakers,
        texts,
        text_lens,
        max_text_lens,
        ref_mels=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(text_lens, max_text_lens)

        output = self.encoder(texts, src_masks)
        
        pred_phn = self.phone_level_emb(self.phoneme_lv_predictor(output))
        output = output + pred_phn
        
        if ref_mels != None:
            utter_out = self.utter_encoder(ref_mels)
            output = output + utter_out.transpose(1,2).repeat(1, output.size(1), 1)
        
        
        speaker_emb = None
        if self.speaker_emb is not None:
            speaker_emb = self.speaker_emb(speakers)
            output = output + speaker_emb.unsqueeze(1).expand(
                -1, max_text_lens, -1
            )

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            None,
            None,
            None,
            None,
            None,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks.cpu(), speaker_emb=speaker_emb)
        output = self.mel_linear(output)
        if self.use_cond_norm:
            output = self.layer_norm(output, speaker_emb)
        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            text_lens,
            mel_lens,
            None,
            pred_phn
        )