import torch
import yaml

from gragod.utils import load_training_data
from models.mtad_gat.model import MTAD_GAT
from models.mtad_gat.predictor import Predictor

DATA_PATH = "data"
PARAMS_FILE = "models/mtad_gat/params.yaml"


def main(params):
    X_train, X_val, X_test, X_labels_train, X_labels_val, X_labels_test = (
        load_training_data(DATA_PATH, normalize=False, replace_anomaly=False)
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(
        "output/mtad_gat/model.pt", map_location=torch.device(device)
    )
    window_size = params["train_params"]["window_size"]
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
    if device == "cuda":
        model = model.to(device)
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


if __name__ == "__main__":
    with open(PARAMS_FILE, "r") as yaml_file:
        params = yaml.safe_load(yaml_file)
    main(params)
