import os
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from omegaconf import OmegaConf

cfg = OmegaConf.load('conf/config.yml')

log_root = Path('multirun')

filename = cfg.catalog.output.summary
logfiles = [
    file for file in
    log_root.glob(f'**/{filename}')]

keyfunc = lambda file: OmegaConf.load(file).rmse 

# find best log file
best_logfile = sorted(logfiles, reverse=False, key= keyfunc)[0]

st.set_page_config(layout="wide")


@st.cache
def load_columns():
    return pd.read_csv(cfg.catalog.clean, index_col=[0], parse_dates=[0]).columns



@st.cache
def load_data():
    columns = load_columns()[1:]

    predict = pd.read_csv(best_logfile.parent / 'output/ECG_data/test/predict.csv', header=None)
    target = pd.read_csv(best_logfile.parent / 'output/ECG_data/test/target.csv', header=None)
    predict = pd.DataFrame(predict.values[:, :len(columns)], columns=columns)
    target = pd.DataFrame(target.values[:,:len(columns)], columns=columns)
    return predict, target




predict, target = load_data()
sid = st.sidebar.selectbox("Select SID", load_columns()[1:])

@st.cache()
def plot_sid(sid):
    fig, ax = plt.subplots(figsize=(20, 10))
    predict, target = load_data()
    predict[sid].plot(ax=ax, color='r')
    target[sid].plot(ax=ax, color='b')
    return fig
    st.pyplot(fig)



fig, ax = plt.subplots(figsize=(20, 10))
predict, target = load_data()
predict[sid].plot(ax=ax, color='r')
target[sid].plot(ax=ax, color='b')
st.pyplot(fig)
