import torch
from torch import nn
import pytorch_lightning as pl
from .adaspeech1 import Adaspeech1
from .loss import Adaspeech1Loss
import matplotlib.pyplot as plt
from utils.model import get_vocoder
from utils.tools import synth_one_sample
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
class PL_model(pl.LightningModule):
    def __init__(self, train_config, preprocess_config, model_config):
        super().__init__()
        self.save_hyperparameters()
        
        self.preprocess_config = preprocess_config
        self.model_config = model_config
        self.train_config = train_config
        
        self.model = Adaspeech1(preprocess_config, model_config, train_config)
        self.loss = Adaspeech1Loss(preprocess_config, train_config)
        self.draw_step = 0
        
    def forward(self, data):
        return self.model(**data, global_step=self.global_step)
    
    def inference(self, data):
        return self.model._inference(**data)
    
    def training_step(self, batch, batch_idx):
        _, inputs = batch
        preds = self.forward(inputs)
        
        losses = self.loss(inputs, preds, self.global_step)
        
        self.log("total_loss",  losses[0], on_epoch=False, on_step=True)
        self.log("mel_loss", losses[1], on_epoch=False, on_step=True)
        self.log("postnet_mel_loss", losses[2], on_epoch=False, on_step=True)
        self.log("pitch_loss", losses[3], on_epoch=False, on_step=True)
        self.log("energy_loss", losses[4], on_epoch=False, on_step=True)
        self.log("duration_loss", losses[5], on_epoch=False, on_step=True)
        self.log("acoustic_loss", losses[6], on_epoch=False, on_step=True)
        return losses[0]

    def validation_step(self, batch, batch_idx):
        meta, inputs = batch
        preds = self.forward(inputs)
        
        losses = self.loss(inputs, preds, self.global_step)
        
        self.log("val_total_loss",  losses[0], on_epoch=True, on_step=False, sync_dist=True)
        self.log("val_mel_loss", losses[1], on_epoch=True, on_step=False, sync_dist=True)
        self.log("val_postnet_mel_loss", losses[2], on_epoch=True, on_step=False, sync_dist=True)
        self.log("val_pitch_loss", losses[3], on_epoch=True, on_step=False, sync_dist=True)
        self.log("val_energy_loss", losses[4], on_epoch=True, on_step=False, sync_dist=True)
        self.log("val_duration_loss", losses[5], on_epoch=True, on_step=False, sync_dist=True)
        self.log("val_acoustic_loss", losses[6], on_epoch=True, on_step=False, sync_dist=True)
        
        return [meta, inputs, preds]
    
    def validation_epoch_end(self, validation_step_outputs):
        if self.draw_step <= self.global_step:
            vocoder = get_vocoder(self.model_config, 'cuda')
            meta, inputs, preds = validation_step_outputs[0]
            fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                meta['ids'][0],
                inputs,
                preds,
                vocoder,
                self.model_config,
                self.preprocess_config,
            )
            
            self.logger.experiment.add_figure(
                "Training/step_{}_{}".format(self.global_step, tag), fig, self.global_step)
            
            self.logger.experiment.add_audio(
                "Training/step_{}_{}_reconstructed".format(self.global_step, tag),
                wav_reconstruction / max(abs(wav_reconstruction)),
                sample_rate=self.preprocess_config["preprocessing"]["audio"]['sampling_rate'],
            )
            self.logger.experiment.add_audio(
                "Training/step_{}_{}_synthesized".format(self.global_step, tag),
                wav_prediction / max(abs(wav_prediction)),
                sample_rate=self.preprocess_config["preprocessing"]["audio"]['sampling_rate'],
            )
            self.draw_step += self.train_config['step']['synth_step']
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            betas=self.train_config["optimizer"]["betas"],
            eps=self.train_config["optimizer"]["eps"],
            weight_decay=self.train_config["optimizer"]["weight_decay"],
            lr=self.train_config["optimizer"]['lr']
        )
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer,
            num_training_steps=self.train_config['step']['total_step'],
            num_warmup_steps=self.train_config['step']['warm_up_step'],
            num_cycles=self.train_config['step']['num_cycle']
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]