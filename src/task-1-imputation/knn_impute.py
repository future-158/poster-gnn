
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from darts import TimeSeries
from darts.models import KalmanFilter
from darts.utils import timeseries_generation as tg
from omegaconf import OmegaConf
import pandas as pd
import joblib
import plotly.express as px

cfg = OmegaConf.load('config.yaml')
df = pd.read_csv(cfg.imputed_path, header=None)

df.columns = list(map(str, df.columns))

# correlation basecd
# (
#     df.corr()
#     .describe(percentiles=[0.99])
#     .T[['mean','min','99%']].plot()
# )

# filter by correlation
df = df.loc[:,~df.corr().mean(axis=0).le(0.6)]

knn_3 = np.argsort(
    df.corr().values,axis=1
)[:,-4:-1]




knn_getter = pd.Series(data = np.array(df.columns)[knn_3].tolist(), index= df.columns).to_dict()

# 110개 그림 그려서 이전, 이후 확인
for sid, ser in df.iteritems():
    nn_sids = knn_getter[sid]
    series = ser.values
    covariates = df.loc[:, nn_sids].values
    
    y_noise = TimeSeries.from_values(
        series
        )
    u = TimeSeries.from_values(
        covariates
        )

    kf = KalmanFilter(dim_x=1)
    kf.fit(y_noise, u)
    y_filtered = kf.filter(y_noise, u)
    # flat_filterd = np.asarray(y_filtered).flatten()

    # plt.figure(figsize=[12, 8])
    # u.plot(label="control")
    # y_noise.plot(color="red", label="Noisy observations", alpha=0.5)
    # y_filtered.plot(color="blue", label="Filtered observations",alpha=0.5)
    # plt.legend()

    pdf = (
        df[[sid, *nn_sids]]
        .assign(
            **{f'{sid}_filtered':  y_filtered.pd_series().values.flatten()}
                        )
    )

    pdf = pd.melt(
        pdf.reset_index(),
        id_vars=['index'], value_vars=pdf.columns
        )

    
    fig = px.line(
        pdf,
        x = 'index',
        y='value',
        color='variable'
                    # hover_data={"date": "|%B %d, %Y"},
                    # title='custom tick labels'
                    )
    # fig.update_xaxes(
    #     dtick="M1",
    #     tickformat="%b\n%Y"
    #     )
    # fig.show()

    
    # fig.write_image(
    #     Path(cfg.fig_dir) / f'{sid}.png'
    # )
    fig.write_html(
        Path(cfg.fig_dir) / f'{sid}.html'
    )


NOISE_DISTANCE = 0.1
SAMPLE_SIZE = 200
np.random.seed(42)

# Prepare the input
u = TimeSeries.from_values(np.heaviside(np.linspace(-5, 10, SAMPLE_SIZE), 0))

# Prepare the output
y = u * TimeSeries.from_values(1 - np.exp(-np.linspace(-5, 10, SAMPLE_SIZE)))

# Add white noise to obtain the observations
noise = tg.gaussian_timeseries(length=SAMPLE_SIZE, std=NOISE_DISTANCE)
y_noise = y + noise

plt.figure(figsize=[12, 8])
u.plot(label="Input")
y.plot(color="gray", label="Output")
y_noise.plot(color="red", label="Noisy observations")
plt.legend()
plt.show()


kf = KalmanFilter(dim_x=1)
kf.fit(y_noise, u)
y_filtered = kf.filter(y_noise, u)

plt.figure(figsize=[12, 8])
u.plot(label="Input")
y.plot(color="gray", label="Output")
y_noise.plot(color="red", label="Noisy observations")
y_filtered.plot(color="blue", label="Filtered observations")
plt.legend()



y_filtered = kf.filter(y_noise, u, num_samples=1000)

plt.figure(figsize=[12, 8])
u.plot(label="Input")
y.plot(color="gray", label="Output")
y_noise.plot(color="red", label="Noisy observations")
y_filtered.plot(color="blue", label="Filtered observations")
plt.legend()



NOISE_DISTANCE = 0.5
SAMPLE_SIZE = 10000
RESIZE_NOISE = 150

# Prepare the drawing
theta = np.radians(np.linspace(360 * 15, 0, SAMPLE_SIZE))
r = theta**2
x_2 = r * np.cos(theta)
y_2 = r * np.sin(theta)

# add white noise (gaussian noise, can be mapped from the random distribution using rand**3)
# and resize to RESIZE_NOISE
x_2_noise = x_2 + (np.random.normal(0, NOISE_DISTANCE, SAMPLE_SIZE) ** 3) * RESIZE_NOISE
y_2_noise = y_2 + (np.random.normal(0, NOISE_DISTANCE, SAMPLE_SIZE) ** 3) * RESIZE_NOISE

plt.figure(figsize=[20, 20])
plt.plot(x_2_noise, y_2_noise, color="red", label="Noisy spiral drawing.")
plt.plot(x_2, y_2, label="Original spiral drawing.")
plt.legend()
plt.show()


# The above drawing can be generated using a second order linear dynamical system without input
#  (decaying to its equilibrium at `(0, 0)`). 
# Therefore we can fit a Kalman filter with `dim_x=2` on the multivariate time series containing both the "x" and "y" components. W
# e see that the Kalman filter does a good job at denoising the spiral drawing.
# 
# Note that we had to adapt the parameter `num_block_rows` for the model fitting to converge.

# In[7]:


kf = KalmanFilter(dim_x=2)
ts = TimeSeries.from_values(x_2_noise).stack(TimeSeries.from_values(y_2_noise))
kf.fit(ts, num_block_rows=50)

filtered_ts = kf.filter(ts).values()
filtered_x = filtered_ts[:, 0]
filtered_y = filtered_ts[:, 1]

plt.figure(figsize=[20, 20])
plt.plot(x_2_noise, y_2_noise, color="red", label="Noisy spiral drawing.")
plt.plot(
    filtered_x, filtered_y, color="blue", linewidth=2, label="Filtered spiral drawing."
)
plt.legend()


# In[ ]:




