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

# %%
train_dataset = Dataset(
    "train.txt", preprocess_config, train_config, sort=True, drop_last=True
)
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=train_dataset.collate_fn,
)
for i, data in enumerate(train_loader):
    meta, inputs = data
    break
# %%
model = PL_model(train_config, preprocess_config, model_config).load_from_checkpoint('./output/ckpt/visual_novel/visual_novel/step=024000.ckpt')
vocoder = get_vocoder(model_config, 'cpu')
# output = model.encoder(inputs['texts'], src_masks)
# %%
def forward(self, data, is_inference):
    data['is_inference'] = is_inference
    return self.model(**data)

model.forward = forward.__get__(model)
prediction = model(inputs, False)
synth_samples(meta, prediction, vocoder, model_config, preprocess_config, './predictions/')
