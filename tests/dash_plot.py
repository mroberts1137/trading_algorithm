import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import random
from flask import Flask

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='live-graph'),
    dcc.Interval(
        id='interval-component',
        interval=1*1000,  # Update every second
        n_intervals=0
    )
])

x_data, y_data = [], []

@app.callback(Output('live-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph_live(n):
    x_data.append(n)
    y_data.append(random.random())
    data = go.Scatter(
        x=list(x_data),
        y=list(y_data),
        mode='lines+markers'
    )
    return {'data': [data], 'layout': go.Layout(xaxis=dict(range=[0, max(x_data)]),
                                                yaxis=dict(range=[min(y_data), max(y_data)]))}

if __name__ == '__main__':
    app.run_server(debug=True)
