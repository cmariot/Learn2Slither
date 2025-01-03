if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    scores = []
    with open("scores.txt", "r") as f:
        for line in f:
            scores.append(int(line))

    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel("Score")
    plt.xlabel("Game")
    plt.show()
