import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import random

# Initialize the application
app = QtWidgets.QApplication([])

# Create the window and plot
win = pg.GraphicsLayoutWidget(show=True)
plot = win.addPlot(title="Real-Time Plot")
curve = plot.plot()

# Data container
data = []
x_data = []


# Update function
def update():
    # Append new x and y data
    x_data.append(len(x_data))  # Increment x value (simulates time or index)
    data.append(random.random())  # Add a new random data point

    # Update the curve with new data
    curve.setData(x_data, data)

    # Limit the visible x-range to the last 100 points
    plot.setXRange(max(0, len(x_data) - 100), len(x_data))

    # Automatically scale the y-axis based on the current data
    plot.enableAutoRange('y', True)


# Set up a timer to call the update function periodically
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(100)  # Update every 1000 milliseconds (1 second)

# Start the Qt event loop
QtWidgets.QApplication.instance().exec_()
