# %%
from model.pl_model import PL_model
from dataset import Dataset
import yaml
from torch.utils.data import DataLoader
from utils.tools import synth_samples
from utils.model import get_vocoder

preprocess_config = yaml.load(
    open("./config/visual_novel.en.add_pause/preprocess.yaml", "r"), Loader=yaml.FullLoader
)
train_config = yaml.load(
    open("./config/visual_novel.en.add_pause/train.yaml", "r"), Loader=yaml.FullLoader
)
model_config = yaml.load(
    open("./config/visual_novel.en.add_pause/model.yaml", "r"), Loader=yaml.FullLoader
)

test_dataset = Dataset(
    "test.txt", preprocess_config, train_config, sort=True, drop_last=False
)
test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=test_dataset.collate_fn,
)
# %%
model = PL_model(train_config, preprocess_config, model_config).load_from_checkpoint("./output/ckpt/visual_novel.en_trim_dur.add_pause/step=040000.ckpt")
vocoder = get_vocoder(model_config, 'cpu')

def forward(self, data, is_inference):
    data['is_inference'] = is_inference
    return self.model(**data)


model.forward = forward.__get__(model)
for i, data in enumerate(test_loader):
    meta, inputs = data
    prediction = model(inputs, False)
    synth_samples(meta, prediction, vocoder, model_config, preprocess_config, './predictions/')
# %%

# %%
