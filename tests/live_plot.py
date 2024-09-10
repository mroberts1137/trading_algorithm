import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Initialize a figure and axis object
fig, ax = plt.subplots()

# Set up plot parameters
x_data, y_data = [], []
line, = ax.plot([], [], lw=2)
ax.set_xlim(0, 10)
ax.set_ylim(-1, 1)

# Function to initialize the plot
def init():
    line.set_data([], [])
    return line,

# Function to update the plot with new data points
def update(frame):
    # Simulate a time series data generation
    x_data.append(frame)
    y_data.append(np.sin(frame))

    # Update the graph with new data
    line.set_data(x_data, y_data)

    # Optionally, adjust the x-limits dynamically if needed
    if frame > 9:
        ax.set_xlim(frame - 9, frame + 1)

    # ax.relim()
    # ax.autoscale_view()

    return line,

# Create animation object (generates new data every 100 ms)
ani = animation.FuncAnimation(fig, update, frames=np.linspace(0, 20, 200), init_func=init, blit=True, interval=100)

plt.show()
