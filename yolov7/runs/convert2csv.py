import os
import json
import pandas as pd

FILE_NAME = "./test/yolov7e6_1280_val/best_predictions.json"
OUTPUT_NAME = "res.csv"

with open(FILE_NAME, "r") as f:
    data = json.load(f)


lst = [""] * 4871
for d in data:
    lst[d["image_id"]] += (
        f"{d['category_id']} {d['score']} " + " ".join(map(str, d["bbox"])) + " "
    )


df = pd.DataFrame()

df["PredictionString"] = lst
df["image_id"] = [f"test/{i:04d}.jpg" for i in range(4871)]

df.to_csv(OUTPUT_NAME, index=False)
