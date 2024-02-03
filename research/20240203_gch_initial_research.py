# %%
import pandas as pd
import start_research

# %%
df_train = pd.read_csv("data/TELCO_data_train.csv")
df_train_labels = pd.read_csv("data/TELCO_labels_train.csv")
df_val = pd.read_csv("data/TELCO_data_val.csv")
df_val_labels = pd.read_csv("data/TELCO_labels_val.csv")
df_test = pd.read_csv("data/TELCO_data_test.csv")
df_test_labels = pd.read_csv("data/TELCO_labels_test.csv")
# %%
df_train
# %%
df_test
# %%
