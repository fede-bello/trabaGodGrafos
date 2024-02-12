# %%
import pandas as pd
import start_research
import torch
import yaml
from torch.utils.data import DataLoader

from gragod.utils import load_training_data
from models.mtad_gat.dataset import SlidingWindowDataset
from models.mtad_gat.model import MTAD_GAT
from models.mtad_gat.predictor import Predictor

# %%
DATA_PATH = "data"
PARAMS_FILE = "models/mtad_gat/params.yaml"
with open(PARAMS_FILE, "r") as yaml_file:
    params = yaml.safe_load(yaml_file)

window_size = params["train_params"]["window_size"]
# %%
X_train, X_val, X_test, X_labels_train, X_labels_val, X_labels_test = (
    load_training_data(DATA_PATH, normalize=False, clean=False)
)
train_dataset = SlidingWindowDataset(X_train, window_size, target_dim=None)
val_dataset = SlidingWindowDataset(X_val, window_size, target_dim=None)
# %%
train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=False)
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
y_pred = torch.Tensor()
y = torch.Tensor()
anomalies_detecetd = 0
anomalied_undetected = 0
i = 0
for idx, (x, y_) in enumerate(train_dataloader):
    with torch.no_grad():
        y_batch, _ = model(x)
    y_pred = y_batch
    y = y_
    error_batch = torch.sqrt((y_pred - y.squeeze()) ** 2)
    mask_batch = X_labels_train[idx * 512 : idx * 512 + 512, ...] == 1.0
    if int(mask_batch.sum()) == 0:
        continue
    error_anomalies = error_batch[mask_batch].mean()
    error_normal = error_batch[~mask_batch].mean()
    print(f"Error for anomalies: {error_anomalies}")
    print(f"Error for normal: {error_normal}")

    if error_anomalies > error_normal:
        anomalies_detecetd += 1
    else:
        anomalied_undetected += 1
    # y_pred = torch.cat((y_pred, y_batch), 0)
    # y = torch.cat((y, y_), 0)
    i += 1
    if i == 1:
        break

print("Anomalies detected: ", anomalies_detecetd)
print("Anomalies undetected: ", anomalied_undetected)

# %%

predictor_params = params["predictor_params"]
predictor = Predictor(
    model,
    window_size,
    n_features,
    predictor_params,
)
predictor.predict_anomalies(
    torch.tensor(X_train), torch.tensor(X_test), None, save_output=True
)

# %%
df = pd.read_pickle("output/mtad_gat/train_output.pkl")

# %%
n_features = X_train.shape[1]
pred_columns = [f"A_Pred_{i}" for i in range(n_features)]
recon_pred_columns = [f"Recon_{i}" for i in range(n_features)]
score_columns = [f"A_Score_{i}" for i in range(n_features)]
scores = df[score_columns]
prediction = df[pred_columns]
y_pred = df[recon_pred_columns].to_numpy()
# %%
prediction
# %%
y_pred.shape
# %%
df
# %%
X_train[window_size:]
# %%
S = scores.to_numpy()
# %%
mask = (X_labels_train == 1.0)[window_size:]
# %%
print(f"Mean score of anomalies: {S[mask].mean()}")
print(f"Mean score of normal: {S[~mask].mean()}")
# %%
th = 2.690941
prediction = S > th
real_value = X_labels_train[window_size:]
print(
    f"Normal value acc: {(~prediction[real_value == 0.0]).sum() / (real_value == 0.0).sum()}"
)
print(f"Recall: {prediction[real_value == 1.0].sum() / (real_value == 1.0).sum()}")
print(f"Precision: {prediction[real_value == 1.0].sum() / prediction.sum()}")
# %%
prediction
# %%
