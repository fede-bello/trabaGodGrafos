# %%
import numpy as np
import pandas as pd
import start_research
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score

from gragod.utils import load_data, load_df, load_training_data

# from gnnad.graphanomaly import GNNAD, eval_metrics
from models.gnnad.model import GNNAD

# %%
slide_win = 300
topk = 6
model = GNNAD(
    threshold_type="max_validation",
    topk=6,
    slide_win=slide_win,
    epoch=1000,
    lr=0.0001,
    early_stop_win=100,
)
# %%
X_train, X_val, X_test, X_labels_train, X_labels_val, X_labels_test = (
    load_training_data("data", normalize=False, replace_anomaly="delete")
)
df_train, df_train_labels, df_val, df_val_labels, df_test, df_test_labels = load_df(
    "data", replace_anomaly="delete"
)
df_train = df_train.drop(columns=["time"])
df_test = df_test.drop(columns=["time"])
df_test_labels = df_test_labels.drop(columns=["time"])
df_train_labels = df_train_labels.drop(columns=["time"])
# %%
# df_train = pd.read_csv(
#     "/Users/gonza/facultad/aagrafos/gnnad/examples/herbert_train.csv"
# )
# df_test = pd.read_csv("/Users/gonza/facultad/aagrafos/gnnad/examples/herbert_test.csv")

# df_test_labels = df_test[
#     [
#         "Unnamed: 0",
#         "anom_1",
#         "anom_2",
#         "anom_3",
#         "anom_4",
#         "anom_5",
#         "anom_6",
#         "anom_7",
#         "anom_8",
#     ]
# ]
# df_test = df_test.drop(
#     columns=[
#         "anom_1",
#         "anom_2",
#         "anom_3",
#         "anom_4",
#         "anom_5",
#         "anom_6",
#         "anom_7",
#         "anom_8",
#     ]
# )
# %%
# model._load_data(df_train, df_train, df_labels_train)
# %%
model.fit(df_train, df_train, df_train_labels)
# %%
y_pred = model.predict_batch(df_train, df_test, df_test_labels)
# %%
y_truth = df_train_labels.to_numpy()[slide_win:, ...]
# %%
error = np.abs(y_truth - y_pred)
# mean = np.mean(error, axis=0).shape
# q75, q25 = np.percentile(error, [75, 25], axis=0)
# iqr = q75 - q25

# error = (error - mean) / iqr
# %%
total_mask = y_truth == 1
print(f"Anomalies mean: {error[total_mask].mean()}")
print(f"Normal mean: {error[~total_mask].mean()}")
# %%
mean_anomalies = []
mean_normal = []
for i in range(error.shape[1]):
    mask = total_mask[:, i]
    print(f"Anomalies mean for feature {i}: {error[mask, i].mean()}")
    print(f"Normal mean for feature {i}: {error[~mask, i].mean()}")
    print(
        f"Threshold for feature {i}: {(error[mask, i].mean() + error[~mask, i].mean()) / 2}"
    )
    mean_anomalies.append(error[mask, i].mean())
    mean_normal.append(error[~mask, i].mean())
thresholds = (np.array(mean_anomalies) + np.array(mean_normal)) / 2
# %%
thresholds = []
for i in range(error.shape[1]):
    real_value = y_truth[:, i]

    # def objective_function(threshold):
    #     prediction = error[:, i] > threshold
    #     return -f1_score(real_value, prediction)
    def objective_function(threshold):
        prediction = error[:, i] > threshold
        f1 = f1_score(real_value, prediction)
        num_predictions = prediction.sum()
        num_anomalies = (real_value == 1.0).sum()
        penalty = abs(num_predictions - num_anomalies)
        return -f1

    result = minimize_scalar(
        objective_function,
        bounds=(error.min(axis=0)[i], error.max(axis=0)[i]),
        method="bounded",
    )
    thresholds.append(result.x)
# %%
prediction = error > thresholds
recalls = {}
precisions = {}
f1_scores = {}

for i in range(error.shape[1]):
    real_value = y_truth[:, i]
    precision = prediction[real_value == 1.0, i].sum() / prediction[:, i].sum()
    recall = prediction[real_value == 1.0, i].sum() / (real_value == 1.0).sum()
    f1 = 2 * (precision * recall) / (precision + recall)
    recalls[i] = recall
    precisions[i] = precision
    f1_scores[i] = f1
for key, value in recalls.items():
    print(
        f"Feature {key} | Precision {precisions[key]:.3} | Recall: {value:.3} | F1: {f1_scores[key]:.3} | Mean Score Anomalies: {mean_anomalies[key]:.3} | Mean Score Normal: {mean_normal[key]:.3} | Number of predictions: {prediction[:, key].sum()} | Number of anomalies: {int(y_truth[:, key].sum())}"
    )
# %%
np.array(list(recalls.values())).mean()

# %%
# -----------------------------------------------------------------------------------
# %%
X_train = df_train.drop(columns=["time"])
X_test = df_test.drop(columns=["time"])
y_test = df_test_labels.drop(columns=["Unnamed: 0"])
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

# %%
feature_list = X_train.columns[
    X_train.columns.str[0] != "_"
].to_list()  # convention is to pass non-features as '_'
assert len(feature_list) == len(set(feature_list))
# %%
fc_struc = {
    ft: [x for x in feature_list if x != ft] for ft in feature_list
}  # fully connected structure

edge_idx_tuples = [
    (feature_list.index(child), feature_list.index(node_name))
    for node_name, node_list in fc_struc.items()
    for child in node_list
]
# %%
fc_edge_idx = [
    [x[0] for x in edge_idx_tuples],
    [x[1] for x in edge_idx_tuples],
]
fc_edge_idx = torch.tensor(fc_edge_idx, dtype=torch.long)


# %%
def parse_data(data, feature_list, labels=None):
    """
    In the case of training data, fill the last column with zeros. This is an
    implicit assumption in the uhnsupervised training case - that the data is
    non-anomalous. For the test data, keep the labels.
    """
    labels = [0] * data.shape[0] if labels is None else labels
    res = data[feature_list].T.values.tolist()
    res.append(labels)
    return res


train_input = parse_data(X_train, feature_list)
test_input = parse_data(X_test, feature_list)
slide_win = 15
slide_stride = 5
cfg = {
    "slide_win": slide_win,
    "slide_stride": slide_stride,
}
# %%
from torch.utils.data import Dataset


class TimeDataset(Dataset):
    """
    A PyTorch dataset class for time series data, to provideadditional functionality for
    processing time series data.

    Attributes
    ----------
    raw_data : list
        A list of raw data
    config : dict
        A dictionary containing the configuration of dataset
    edge_index : np.ndarray
        Edge index of the dataset
    mode : str
        The mode of dataset, either 'train' or 'test'
    x : torch.Tensor
        Feature data
    y : torch.Tensor
        Target data
    labels : torch.Tensor
        Anomaly labels of the data
    """

    def __init__(self, raw_data, edge_index, mode="train", config=None):
        self.raw_data = raw_data
        self.config = config
        self.edge_index = edge_index
        self.mode = mode

        # to tensor
        data = torch.tensor(raw_data[:-1]).double()
        labels = torch.tensor(raw_data[-1]).double()
        self.x, self.y, self.labels = self.process(data, labels)

    def __len__(self):
        return len(self.x)

    def process(self, data, labels):
        x_arr, y_arr, labels_arr = [], [], []
        slide_win, slide_stride = self.config["slide_win"], self.config["slide_stride"]
        is_train = self.mode == "train"
        total_time_len = data.shape[1]

        for i in (
            range(slide_win, total_time_len, slide_stride)
            if is_train
            else range(slide_win, total_time_len)
        ):
            ft = data[:, i - slide_win : i]
            tar = data[:, i]
            x_arr.append(ft)
            y_arr.append(tar)
            labels_arr.append(labels[i])

        x = torch.stack(x_arr).contiguous()
        y = torch.stack(y_arr).contiguous()
        labels = torch.Tensor(labels_arr).contiguous()

        return x, y, labels

    def __getitem__(self, idx):
        feature = self.x[idx].double()
        y = self.y[idx].double()
        edge_index = self.edge_index.long()
        label = self.labels[idx].double()

        return feature, y, label, edge_index


# %%
train_dataset = TimeDataset(train_input, fc_edge_idx, mode="train", config=cfg)
test_dataset = TimeDataset(test_input, fc_edge_idx, mode="test", config=cfg)

# %%
