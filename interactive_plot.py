import matplotlib.pyplot as plt
import random
import time


if __name__ == "__main__":
    plt.ion()
    figure, ax = plt.subplots()
    x = []
    y = []
    line, = ax.plot(x, y)

    for i in range(100):
        x.append(i)
        y.append(random.randint(0, 10))
        line.set_xdata(x)
        line.set_ydata(y)
        ax.relim()
        # Rescale the y-axis to fit the new data
        ax.autoscale_view()
        figure.canvas.draw()
        figure.canvas.flush_events()
        plt.pause(0.1)

        time.sleep(0.5)
    plt.ioff()
