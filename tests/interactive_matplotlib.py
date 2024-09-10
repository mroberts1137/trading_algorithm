import matplotlib.pyplot as plt
import time
import random

plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
x_data, y_data = [], []
line, = ax.plot(x_data, y_data)

for t in range(100):
    x_data.append(t)
    y_data.append(random.random())
    line.set_xdata(x_data)
    line.set_ydata(y_data)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(1)  # Simulate delay for new data
