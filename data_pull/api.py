import requests
import json
import pandas as pd
import time

js = []
keywords = ["environmental", "sciences"]

st = time.process_time()
for num in range(50):
    r = requests.get(f"https://www.osti.gov/api/v1/records?q={"+".join(keywords)}/page={num}")
    records = r.json()
    js.extend(records)
    print(len(js))

df = pd.DataFrame(js)
df = pd.DataFrame.from_dict(js)
df = pd.DataFrame.from_records(js)
df.to_csv("test.csv")


# get execution time
et = time.process_time()
res = et - st
print('CPU Execution time:', res, 'seconds')