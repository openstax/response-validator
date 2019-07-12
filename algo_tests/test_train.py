import pandas as pd
import time
import requests
BASE_URL = "http://127.0.0.1:5000/train"
DATAPATHS = "./data/expert_grader_valid_100.csv"
df = pd.read_csv(DATAPATHS)
df = df.rename(columns = {"valid":"valid_label"})
df_json = df.to_json()
r = requests.post(BASE_URL, json={"response_df": df_json})
print(r.json()["intercept"])