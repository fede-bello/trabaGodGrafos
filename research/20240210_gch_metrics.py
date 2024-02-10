# %%
import start_research
import torch
import yaml

from gragod.utils import load_training_data
from models.mtad_gat.dataset import SlidingWindowDataset
from models.mtad_gat.model import MTAD_GAT

# %%
DATA_PATH = "data"
PARAMS_FILE = "models/mtad_gat/params.yaml"
with open(PARAMS_FILE, "r") as yaml_file:
    params = yaml.safe_load(yaml_file)

window_size = params["train_params"]["window_size"]
# %%
X_train, X_val, X_test, X_labels_train, X_labels_val, X_labels_test = (
    load_training_data(DATA_PATH, normalize=True, clean=False)
)
train_dataset = SlidingWindowDataset(X_train, window_size, target_dim=None)
val_dataset = SlidingWindowDataset(X_val, window_size, target_dim=None)
# %%
state_dict = torch.load("output/mtad_gat/model.pt")
# %%
n_features = X_train.shape[1]
out_dim = X_train.shape[1]

model = MTAD_GAT(
    n_features,
    window_size,
    out_dim,
    dropout=params["train_params"]["dropout"],
    **params["model_params"],
)
model.load_state_dict(state_dict)
model.eval()
# %%
X = next(iter(train_dataset))
# %%
y_pred = model(X[0].unsqueeze(0))
# %%
y_pred[1].shape
# %%
X[0] - y_pred[1]
# %%
X[0]
# %%
y_pred[1]
# %%
X_train, X_val, X_test, X_labels_train, X_labels_val, X_labels_test = (
    load_training_data(DATA_PATH, normalize=False, clean=False)
)
# %%
