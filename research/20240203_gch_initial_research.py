# %%
import os

import pandas as pd
import start_research  # noqa
import temporian as tp

# %%
DATA_PATH = "data"


def load_df(base_path):
    df_train = pd.read_csv(os.path.join(base_path, "TELCO_data_train.csv"))
    df_train_labels = pd.read_csv(os.path.join(base_path, "TELCO_labels_train.csv"))
    df_val = pd.read_csv(os.path.join(base_path, "TELCO_data_val.csv"))
    df_val_labels = pd.read_csv(os.path.join(base_path, "TELCO_labels_val.csv"))
    df_test = pd.read_csv(os.path.join(base_path, "TELCO_data_test.csv"))
    df_test_labels = pd.read_csv(os.path.join(base_path, "TELCO_labels_test.csv"))

    return df_train, df_train_labels, df_val, df_val_labels, df_test, df_test_labels


def load_tp(base_path):
    es_train = tp.from_csv(
        os.path.join(base_path, "TELCO_data_train.csv"), timestamps="time"
    )
    es_label_train = tp.from_csv(
        os.path.join(base_path, "TELCO_labels_train.csv"), timestamps="time"
    )
    es_val = tp.from_csv(
        os.path.join(base_path, "TELCO_data_val.csv"), timestamps="time"
    )
    es_label_val = tp.from_csv(
        os.path.join(base_path, "TELCO_labels_val.csv"), timestamps="time"
    )
    es_test = tp.from_csv(
        os.path.join(base_path, "TELCO_data_test.csv"), timestamps="time"
    )
    es_label_test = tp.from_csv(
        os.path.join(base_path, "TELCO_labels_test.csv"), timestamps="time"
    )

    return es_train, es_label_train, es_val, es_label_val, es_test, es_label_test


# %%
df_train, df_train_labels, df_val, df_val_labels, df_test, df_test_labels = load_df(
    DATA_PATH
)
es_train, es_label_train, es_val, es_label_val, es_test, es_label_test = load_tp(
    DATA_PATH
)
# %%
a = es_train["TS1"]
plot = a.simple_moving_average(tp.duration.seconds(10))
plot.plot()
# %%
es_train.plot()
# %%
a

# %%
df_train[:100][(df_train_labels["TS1"][:100] == 1.0).values]["TS1"].mean()
# %%
df_train[:100][(df_train_labels["TS1"][:100] == 0.0).values]["TS1"].mean()
# %%
df_train["TS1"].std()
# %%
