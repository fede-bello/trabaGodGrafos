# %%
import numpy as np
import pandas as pd
import start_research  # noqa
import torch
import yaml
from sklearn.impute import SimpleImputer
from torch import nn
from torch.utils.data import DataLoader, Dataset

from gragod.utils import load_data, load_df, load_training_data, normalize_data
from models.mtad_gat.dataset import SlidingWindowDataset
from models.mtad_gat.model import MTAD_GAT
from models.mtad_gat.predictor import Predictor
from models.mtad_gat.trainer import Trainer

DATA_PATH = "data"
PARAMS_FILE = "models/mtad_gat/params.yaml"
with open(PARAMS_FILE, "r") as yaml_file:
    params = yaml.safe_load(yaml_file)

# %%
X_train, X_val, X_test, X_labels_train, X_labels_val, X_labels_test = (
    load_training_data(DATA_PATH, normalize=True, clean=True)
)
# %%
window_size = params["train_params"]["window_size"]
train_dataset = SlidingWindowDataset(X_train, window_size, target_dim=None)
val_dataset = SlidingWindowDataset(X_val, window_size, target_dim=None)
print(train_dataset[0][0].shape)
# %%
batch_size = params["train_params"]["batch_size"]
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
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
# %%
lr = 0.001
target_dims = None
n_epochs = 1
init_lr = 0.001
use_cuda = False
save_path = "output/mtad_gat"
print_every = 1
log_tensorboard = True
args_summary = "args_summary"
log_dir = "output/mtad_gat"

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
forecast_criterion = nn.MSELoss()
recon_criterion = nn.MSELoss()

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    n_features=n_features,
    target_dims=target_dims,
    forecast_criterion=forecast_criterion,
    recon_criterion=recon_criterion,
    **params["train_params"],
)
trainer.fit(train_loader, val_loader)
# %%
predictor_params = params["predictor_params"]


# parser = get_parser()
# args = parser.parse_args()

# dataset = args.dataset
# window_size = 10
# spec_res = args.spec_res
# normalize = args.normalize
# n_epochs = args.epochs
# batch_size = args.bs
# init_lr = args.init_lr
# val_split = args.val_split
# shuffle_dataset = args.shuffle_dataset
# use_cuda = args.use_cuda
# print_every = args.print_every
# log_tensorboard = args.log_tensorboard
# group_index = args.group[0]
# index = args.group[2:]
# args_summary = str(args.__dict__)
# print(args_summary)
# # %%
# level_q_dict = {
#     "SMAP": (0.90, 0.005),
#     "MSL": (0.90, 0.001),
#     "SMD-1": (0.9950, 0.001),
#     "SMD-2": (0.9925, 0.001),
#     "SMD-3": (0.9999, 0.001),
#     "custom": (0.9999, 0.001),
# }
# reg_level = 1
# key = "SMD-" + args.group[0] if args.dataset == "SMD" else args.dataset
# level, q = level_q_dict[key]
X_train, X_val, X_test, X_labels_train, X_labels_val, X_labels_test = load_data(
    DATA_PATH
)
X_train, _ = normalize_data(X_train)
X_val, _ = normalize_data(X_val)
X_test, _ = normalize_data(X_test)
with open(PARAMS_FILE, "r") as yaml_file:
    prediction_args = yaml.safe_load(yaml_file)["predictor_params"]

# best_model = trainer.model
best_model = model
predictor = Predictor(
    best_model,
    window_size,
    n_features,
    prediction_args,
)

# %%
predictor.predict_anomalies(
    torch.tensor(X_train), torch.tensor(X_test), None, save_output=True
)
# %%
import pandas as pd


def load_metrics(mode):
    df = pd.read_pickle(f"first_prediction/{mode}_output.pkl")
    y_pred = df["A_Pred_Global"]
    y_pred = y_pred.to_numpy()

    df_train, df_train_labels, df_val, df_val_labels, df_test, df_test_labels = load_df(
        "data"
    )

    if mode == "train":
        df = df_train_labels
    elif mode == "val":
        df = df_val_labels
    elif mode == "test":
        df = df_test_labels
    df = df[:-10]
    y = (df == 1.0).any(axis=1).astype(float)

    print((y == 1.0).sum())
    y = y.to_numpy()

    # print(y.shape, y_pred.shape)
    anomalies_indices = np.where(y == 1.0)[0]
    print(len(anomalies_indices))
    print(
        (y[anomalies_indices] == y_pred[anomalies_indices]).sum()
        / len(anomalies_indices)
    )

    return y[anomalies_indices], y_pred[anomalies_indices], y, y_pred


*_, y_train, y_train_pred = load_metrics("train")
# %%
y_train
# %%
y_train_pred

# %%
import temporian as tp

tp.from_pandas(
    pd.DataFrame(
        {"val": y_train_pred, "timestamp": [i for i in range(len(y_train_pred))]}
    )
).plot()
# %%

tp.from_pandas(
    pd.DataFrame({"val": y_train, "timestamp": [i for i in range(len(y_train))]})
).plot()
# %%
model_path = "output/mtad_gat/model.pt"
state_dict = torch.load(model_path)

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

# %%
x, y = next(iter(train_loader))
# %%
y_pred = model(x)
# %%
len(y_pred)
# %%
print(y_pred[0].shape)
print(y_pred[1].shape)
# %%
mode = "train"
df = pd.read_pickle(f"output/mtad_gat/{mode}_output.pkl")
# %%
X_train, X_val, X_test, X_train_labels, X_labels_val, X_labels_test = (
    load_training_data(DATA_PATH, normalize=True, clean=False)
)
n_features = X_train.shape[1]
pred_columns = [f"A_Pred_{i}" for i in range(n_features)]
recon_pred_columns = [f"Recon_{i}" for i in range(n_features)]
prediction = df[pred_columns]
y_pred = df[recon_pred_columns].to_numpy()
# %%
prediction.sum()
# %%
mask = X_train_labels == 1.0
pred_anomalies = torch.Tensor(y_pred)[mask[20:, :]]
# %%
df_train, df_train_labels, df_val, df_val_labels, df_test, df_test_labels = load_df(
    DATA_PATH
)
# %%
df_train_labels.sum()
# %%
anomalies = torch.Tensor(X_train[20:, :])[mask[20:, :]]
# %%
np.sqrt((pred_anomalies - anomalies) ** 2).sum() / len(pred_anomalies)
# %%

np.sqrt((y_pred - X_train[20:].numpy()) ** 2).sum() / len(y_pred)
# %%
np.sqrt((y_pred - X_train[20:]) ** 2).shape

# %%
X_train_clean, *_ = load_training_data(DATA_PATH, normalize=True, clean=True)
# %%
x, y = next(iter(train_loader))
# %%
y_pred = model(x)
# %%
from gragod.utils import load_params, load_training_data

X_train, X_val, *_ = load_training_data(
    DATA_PATH, normalize=False, replace_anomaly="delete"
)

# Create dataloaders
window_size = params["train_params"]["window_size"]
batch_size = params["train_params"]["batch_size"]

train_dataset = SlidingWindowDataset(X_train, window_size, target_dim=None)
val_dataset = SlidingWindowDataset(X_val, window_size, target_dim=None)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Create model
n_features = X_train.shape[1]
out_dim = X_train.shape[1]


params = load_params(PARAMS_FILE, type="yaml")
model = MTAD_GAT(
    n_features,
    window_size,
    out_dim,
    dropout=params["train_params"]["dropout"],
    **params["model_params"],
)
if params["train_params"]["weights_path"]:
    state_dict = torch.load(
        params["train_params"]["weights_path"], map_location=torch.device("cpu")
    )
    model.load_state_dict(state_dict)
# %%

# %%
