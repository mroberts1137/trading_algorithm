'''
Run> bokeh serve --show bokeh_plot.py
'''

from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource
from random import random
from bokeh.driving import linear

# Data source
source = ColumnDataSource(data=dict(x=[], y=[]))

# Create a plot
p = figure(title="Live Data Stream", x_axis_label='Time', y_axis_label='Value')
p.line('x', 'y', source=source)

# Update function
@linear()
def update(step):
    new_data = dict(x=[step], y=[random()])
    source.stream(new_data, rollover=200)

# Add to document
curdoc().add_root(p)
curdoc().add_periodic_callback(update, 1000)
