import os
from datetime import datetime
from pathlib import Path
from sys import api_version

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from omegaconf import OmegaConf

cfg = OmegaConf.load("conf/config.yml")

log_root = Path("multirun")
filename = cfg.catalog.output.summary
log_files = list(log_root.glob(f"**/{filename}"))

logs = [{**OmegaConf.load(file), "file": file} for file in log_files]

logs = [
    log for log in logs if log.get("api_version") == cfg.api_version and "rmse" in log
]

keyfunc = lambda log: log["rmse"]
best_logfile = sorted(logs, reverse=False, key=keyfunc)[0]["file"]
st.set_page_config(layout="wide")


@st.cache
def load_columns():
    return pd.read_csv(cfg.catalog.clean, index_col=[0], parse_dates=[0]).columns


@st.cache
def load_data():
    columns = load_columns()[1:]

    predict = pd.read_csv(
        best_logfile.parent / f"output/{cfg.args.dataset}/test/predict.csv", header=None
    )
    target = pd.read_csv(
        best_logfile.parent / f"output/{cfg.args.dataset}/test/target.csv", header=None
    )
    predict = pd.DataFrame(predict.values[:, : len(columns)], columns=columns)
    target = pd.DataFrame(target.values[:, : len(columns)], columns=columns)
    return predict, target


predict, target = load_data()
sid = st.sidebar.selectbox("Select SID", load_columns()[1:])


@st.cache()
def plot_sid(sid):
    fig, ax = plt.subplots(figsize=(20, 10))
    predict, target = load_data()
    predict[sid].plot(ax=ax, color="r")
    target[sid].plot(ax=ax, color="b")
    return fig
    st.pyplot(fig)


fig, ax = plt.subplots(figsize=(20, 10))
predict, target = load_data()
predict[sid].plot(ax=ax, color="r")
target[sid].plot(ax=ax, color="b")
st.pyplot(fig)
