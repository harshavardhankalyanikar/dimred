import pandas as pd
import numpy as np

def load_and_clean():
    df = pd.read_csv("data/CC_GENERAL.csv")
    df.drop(columns=["CUST_ID"], inplace=True)
    df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].median(), inplace=True)
    df["MINIMUM_PAYMENTS"].fillna(df["MINIMUM_PAYMENTS"].median(), inplace=True)
    return df
