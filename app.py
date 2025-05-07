#!/usr/bin/env python3
# coding: utf-8

import os
import pandas as pd
import numpy as np

# sklearn for training
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Dash & Plotly for the dashboard
from dash import Dash, html, dcc, Input, Output
import plotly.express as px

# 1) LOAD & PREPROCESS (sample first 5k rows)
energy_df  = pd.read_csv('energy_dataset.csv', nrows=5000)
weather_df = pd.read_csv('weather_features.csv', nrows=5000)

energy_df['time']    = pd.to_datetime(energy_df['time'], utc=True)
weather_df['dt_iso'] = pd.to_datetime(weather_df['dt_iso'], utc=True)

weather_numeric = weather_df.drop(columns=[
    'city_name','weather_main','weather_description','weather_icon'
])
weather_avg = weather_numeric.groupby('dt_iso').mean().reset_index()

df = pd.merge(
    energy_df, weather_avg,
    left_on='time', right_on='dt_iso', how='inner'
)

needed = [
    'forecast solar day ahead','forecast wind onshore day ahead',
    'total load forecast','price actual','total load actual',
    'temp','humidity','pressure','wind_speed','clouds_all'
]
df = df.dropna(subset=needed)

# time features
df['hour']      = df['time'].dt.hour
df['dayofweek'] = df['time'].dt.dayofweek

# lag features
df['price_lag1'] = df['price actual'].shift(1)
df['load_lag1']  = df['total load actual'].shift(1)
df = df.dropna(subset=['price_lag1','load_lag1'])

# split X/y
features = [
    'forecast solar day ahead','forecast wind onshore day ahead',
    'total load forecast','temp','humidity','pressure',
    'wind_speed','clouds_all','hour','dayofweek',
    'price_lag1','load_lag1'
]
X = df[features]
y = df['price actual']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# 2) TRAIN
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# prepare a single DataFrame for plotting
test_df = X_test.copy()
test_df['actual'] = y_test.values
test_df['pred']   = model.predict(X_test)
test_df['time']   = df.loc[y_test.index, 'time']

# 3) DASH SETUP
app    = Dash(__name__)
server = app.server
app.title = "Energy Price Dashboard"

app.layout = html.Div([
    html.H1("🔌 Energy Price: Actual vs Predicted"),
    dcc.Dropdown(
        id='series-picker',
        options=[
            {'label': 'Show Actual',    'value': 'actual'},
            {'label': 'Show Predicted', 'value': 'pred'},
            {'label': 'Both',           'value': 'both'},
        ],
        value='both',
        clearable=False,
        style={'width':'250px'}
    ),
    dcc.Graph(id='time-series')
])

@app.callback(
    Output('time-series', 'figure'),
    Input('series-picker', 'value')
)
def update_figure(chosen):
    df_plot = test_df.copy()
    if chosen == 'actual':
        y_cols = ['actual']
    elif chosen == 'pred':
        y_cols = ['pred']
    else:
        y_cols = ['actual','pred']

    fig = px.line(
        df_plot, x='time', y=y_cols,
        labels={'value':'Price','time':'Time','variable':'Series'},
        title="Energy Price Over Time"
    )
    return fig

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run_server(
        host='0.0.0.0',
        port=port,
        debug=False,
        use_reloader=False
    )





