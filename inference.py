# %%
from model.pl_model import PL_model
from dataset import Dataset
import yaml
from torch.utils.data import DataLoader
from utils.tools import synth_samples
from utils.model import get_vocoder

preprocess_config = yaml.load(
    open("./config/visual_novel/preprocess.yaml", "r"), Loader=yaml.FullLoader
)
train_config = yaml.load(
    open("./config/visual_novel/train.yaml", "r"), Loader=yaml.FullLoader
)
model_config = yaml.load(
    open("./config/visual_novel/model.yaml", "r"), Loader=yaml.FullLoader
)

val_dataset = Dataset(
    "val.txt", preprocess_config, train_config, sort=True, drop_last=True
)
val_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=train_dataset.collate_fn,
)
for i, data in enumerate(train_loader):
    meta, inputs = data
    break

model = PL_model(train_config, preprocess_config, model_config).load_from_checkpoint("Your checkpoints")
vocoder = get_vocoder(model_config, 'cpu')

def forward(self, data, is_inference):
    data['is_inference'] = is_inference
    return self.model(**data)

model.forward = forward.__get__(model)
prediction = model(inputs, False)
synth_samples(meta, prediction, vocoder, model_config, preprocess_config, './predictions/')
