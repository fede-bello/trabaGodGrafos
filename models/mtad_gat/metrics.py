import numpy as np
import pandas as pd
import yaml

from gragod.utils import load_training_data

DATA_PATH = "data"
PARAMS_FILE = "models/mtad_gat/params.yaml"


def main(params):

    window_size = params["train_params"]["window_size"]

    X_train, X_val, X_test, X_labels_train, X_labels_val, X_labels_test = (
        load_training_data(DATA_PATH, normalize=False, replace_anomaly=False)
    )

    n_features = X_train.shape[1]

    df = pd.read_pickle("output/mtad_gat/train_output.pkl")

    n_features = X_train.shape[1]
    pred_columns = [f"A_Pred_{i}" for i in range(n_features)]
    recon_pred_columns = [f"Recon_{i}" for i in range(n_features)]
    score_columns = [f"A_Score_{i}" for i in range(n_features)]
    scores = df[score_columns]
    prediction = df[pred_columns]
    y_pred = df[recon_pred_columns].to_numpy()
    thresholds = df[[f"Thresh_{i}" for i in range(n_features)]].iloc[0].to_numpy()

    S = scores.to_numpy()
    mask = (X_labels_train == 1.0)[window_size:]

    mean_anomalies = []
    mean_normal = []
    for i in range(n_features):
        print(f"Mean score of anomalies for feature {i}: {S[:, i][mask[:,i]].mean()}")
        print(f"Mean score of normal for feature {i}: {S[:, i][~mask[:,i]].mean()}")
        mean_anomalies.append(S[:, i][mask[:, i]].mean())
        mean_normal.append(S[:, i][~mask[:, i]].mean())
    mean_anomalies = np.array(mean_anomalies)
    mean_normal = np.array(mean_normal)

    thresholds = (mean_anomalies + mean_normal) / 2
    for i in range(n_features):
        # print(f"Threshold for feature {i}: {thresholds[i]}")
        real_value = X_labels_train[window_size:, i]
        prediction = S[:, i] > thresholds[i]
        print(
            f"Precision for feature {i}: {prediction[real_value == 1.0].sum() / prediction.sum()}"
        )
        print(
            f"Recall for feature {i}: {prediction[real_value == 1.0].sum() / (real_value == 1.0).sum()}"
        )

    prediction = S > thresholds

    real_value = X_labels_train[window_size:]
    print(
        f"Normal value acc: {(~prediction[real_value == 0.0]).sum() / (real_value == 0.0).sum()}"
    )
    print(f"Recall: {prediction[real_value == 1.0].sum() / (real_value == 1.0).sum()}")
    print(f"Precision: {prediction[real_value == 1.0].sum() / prediction.sum()}")

    print(f"Predictions per feature: {prediction.sum(axis=0)}")
    print(f"Real values per feature: {real_value.sum(axis=0)}")


if __name__ == "__main__":
    with open(PARAMS_FILE, "r") as yaml_file:
        params = yaml.safe_load(yaml_file)
    main(params)
