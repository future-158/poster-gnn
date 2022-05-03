from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from omegaconf import OmegaConf

cfg = OmegaConf.load("conf/config.yml")
st.set_page_config(layout="wide")


@st.cache
def load_data():
    source = cfg.catalogue.impute
    return pd.read_csv(source, parse_dates=["timestamp"], index_col=["timestamp"])


sid = st.sidebar.selectbox("Select SID", load_data().columns)

fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(load_data().loc[:, sid])
st.pyplot(fig)
