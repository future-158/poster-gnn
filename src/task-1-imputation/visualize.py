import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

exp_name = 'plt_2'

fig_dir = Path('fig') / exp_name
fig_li = [x for x in 
fig_dir.glob('*.joblib')]


st.set_page_config(layout = "wide")
sid = st.sidebar.selectbox(
    'Select page',
  [x.stem for x in fig_li]
  )

file = fig_dir /  f'{sid}.joblib'
fig = joblib.load(file)
st.pyplot(fig)