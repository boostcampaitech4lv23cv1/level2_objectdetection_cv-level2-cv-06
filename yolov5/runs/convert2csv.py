import os
import json
import pandas as pd
import argparse
import glob
def convert2csv(file):
    
    path=f"./runs/val/{file}"    
    print(path)
    FILE_NAME = glob.glob(f'{path}/*.json')[0]
    OUTPUT_NAME = f"{file}.csv"

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
    print(f"Saved {OUTPUT_NAME}")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default=None, help='file name')
    args=parser.parse_args()
    
    convert2csv(args.file)
