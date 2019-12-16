import glob
from datetime import datetime

import pandas as pd


def main():
    buffer = []
    for filepath in glob.glob("outputs/*.csv"):
        df_tmp = pd.read_csv(filepath)
        if "timestamp" not in df_tmp.columns:
            df_tmp["timestamp"] = int(datetime(2019, 11, 21).timestamp())
        if "similarity_raw" in df_tmp.columns:
            del df_tmp["similarity_raw"]
        buffer.append(df_tmp)
    df_final = pd.concat(buffer, axis=0, ignore_index=True)
    df_final.to_csv("annotated.csv", index=False)


if __name__ == "__main__":
    main()
