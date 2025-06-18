import pandas as pd
import numpy as np
from xgboost import XGBRegressor, DMatrix
import ta

# Load and train
df_train = pd.read_csv("btc_data_train_labeled_features.csv")
X_train = df_train.drop(columns=["Score"])
y_train = df_train["Score"]

model = XGBRegressor(
    n_estimators=1000,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
model.fit(X_train, y_train)

# Save the model as a binary file
model.save_model("btc_xgb_model.bin")

