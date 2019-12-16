import os
import uuid
import math
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import datetime

import uvicorn
import numpy as np
import pandas as pd
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from fastapi import FastAPI
from pydantic import BaseModel

PORT = int(os.environ.get("PORT", "8666"))
OUTPUT_DIR = Path("outputs/")
APP = FastAPI()
ORIGINS = [
    "*"
]
APP.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
APP.add_middleware(
    SessionMiddleware,
    secret_key="asdjlksd281"
)
DATASET: pd.DataFrame = pd.DataFrame()
PAIRS_PER_PAGE = 25
GLOBAL_CACHE: Dict = {}


class BatchForAnnotation(BaseModel):
    page: int
    pairs: List[Tuple[int, str, str, float]] = []


class BatchAnnotated(BaseModel):
    page: int
    pairs: List[Tuple[int, float]] = []


class SubmitResult(BaseModel):
    success: bool
    overwrite: bool
    message: str


@APP.get("/")
def read_root():
    return {"Hello": "World"}


def find_page_to_annotate(cache_entry: Dict):
    for page in range(int(math.ceil(len(DATASET) / PAIRS_PER_PAGE))):
        if page not in cache_entry["submitted"]:
            return page
    return None


@APP.get("/batch/", response_model=BatchForAnnotation)
def get_batch(request: Request):
    if not request.session or request.session["uid"] not in GLOBAL_CACHE:
        print("Creating new indices...")
        request.session["uid"] = str(uuid.uuid4())
        indices = np.arange(len(DATASET))
        np.random.shuffle(indices)
        GLOBAL_CACHE[request.session["uid"]] = {
            "indices": indices,
            "submitted": {}
        }
    cache_entry = GLOBAL_CACHE[request.session["uid"]]
    page = find_page_to_annotate(cache_entry)
    if page is None:
        return BatchForAnnotation(
            page=-1,
            pairs=[]
        )
    batch = DATASET.loc[
        cache_entry["indices"][
            page*PAIRS_PER_PAGE:(page+1)*PAIRS_PER_PAGE
        ]
    ]
    pairs = list(batch[
        ["text_1", "text_2", "similarity"]
    ].itertuples(index=True, name=None))
    print(f"Page: {page} Items: {len(pairs)}")
    return BatchForAnnotation(
        page=page,
        pairs=pairs
    )


@APP.post("/batch/", response_model=SubmitResult)
def submit_batch(batch: BatchAnnotated, request: Request):
    if (
        not request.session.get("uid") or
        not request.session.get("uid") in GLOBAL_CACHE
    ):
        return SubmitResult(
            success=False,
            overwrite=False,
            message="You haven't fetched any batches yet."
        )
    page = batch.page
    batch_orig = DATASET.loc[
        GLOBAL_CACHE[request.session["uid"]]["indices"][
            page*PAIRS_PER_PAGE:(page+1)*PAIRS_PER_PAGE
        ]
    ].copy()
    indices, similarities = list(zip(*batch.pairs))
    if set(indices) != set(batch_orig.index.tolist()):
        return SubmitResult(
            success=False,
            overwrite=False,
            message="Indices don't match!"
        )
    batch_orig.loc[indices, "similarity"] = similarities
    batch_orig["timestamp"] = int(datetime.now().timestamp())
    overwrite = False
    if page in GLOBAL_CACHE[request.session["uid"]]["submitted"]:
        output_path = GLOBAL_CACHE[request.session["uid"]]["submitted"][page]
        overwrite = True
    else:
        output_path = OUTPUT_DIR / \
            f"{datetime.now().strftime('%Y%m%d_%H%M')}_{page}.csv"
        GLOBAL_CACHE[request.session["uid"]]["submitted"][page] = output_path
    batch_orig.to_csv(output_path, index=False)
    return SubmitResult(
        success=True,
        overwrite=overwrite,
        message=""
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--data-path', type=str, default="data/dataset.csv")
    args = parser.parse_args()
    DATASET = pd.read_csv(args.data_path)
    print(f"Listening to port {PORT}")
    uvicorn.run(APP, host='0.0.0.0', port=PORT)
