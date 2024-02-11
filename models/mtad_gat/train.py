import torch
from torch import nn
from torch.utils.data import DataLoader

from gragod.utils import load_params, load_training_data
from models.mtad_gat.dataset import SlidingWindowDataset
from models.mtad_gat.model import MTAD_GAT
from models.mtad_gat.predictor import Predictor
from models.mtad_gat.trainer import Trainer

DATA_PATH = "data"
PARAMS_FILE = "models/mtad_gat/params.yaml"


def main(params):
    # Load data
    X_train, X_val, X_test, *_ = load_training_data(
        DATA_PATH, normalize=True, clean=True
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

    model = MTAD_GAT(
        n_features,
        window_size,
        out_dim,
        dropout=params["train_params"]["dropout"],
        **params["model_params"],
    )

    # lr = params["train_params"]["lr"]
    # target_dims = params["train_params"]["target_dims"]
    # n_epochs = params["train_params"]["n_epochs"]
    # init_lr = params["train_params"]["init_lr"]
    # use_cuda = params["train_params"]["use_cuda"]
    # save_path = params["train_params"]["save_path"]
    # print_every = params["train_params"]["print_every"]
    # log_tensorboard = params["train_params"]["log_tensorboard"]
    # args_summary = params["train_params"]["args_summary"]
    # log_dir = params["train_params"]["log_dir"]

    # Create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=params["train_params"]["lr"])
    forecast_criterion = nn.MSELoss()
    recon_criterion = nn.MSELoss()

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        n_features=n_features,
        forecast_criterion=forecast_criterion,
        recon_criterion=recon_criterion,
        **params["train_params"],
    )

    # Train model
    trainer.fit(train_loader, val_loader)

    # Create predictor
    predictor_params = params["predictor_params"]
    X_train, X_val, X_test, *_ = load_training_data(
        DATA_PATH, normalize=False, clean=False
    )
    predictor = Predictor(
        model=model,
        window_size=window_size,
        n_features=n_features,
        pred_args=predictor_params,
    )
    predictor.predict_anomalies(
        torch.tensor(X_train), torch.tensor(X_test), None, save_output=True
    )

    # TODO: Save model and prediction in the proper folder


if __name__ == "__main__":
    params = load_params(PARAMS_FILE, type="yaml")

    main(params)
