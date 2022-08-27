import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import choices

def plotNumHeads(run = 2500):
  """ Plots the expected number of heads in 2 tosses of fair coin.

  Using live graph features from matplotlib.
  """
  def expectedHeads(n):
    sumHeads = 0
    for x in range(1, n + 1):
      z = choices(["H", "T"], k = 2)
      sumHeads += z.count("H")
    return (sumHeads/(n))

  y_np = np.array([expectedHeads(i) for i in range(1, run)])
  x_np = np.array(list(range(1,run)))

  sizeI = np.empty(1); sizeI.fill(1)
  colors = np.random.choice(["r", "g", "b"], size=10)

  fig = plt.figure()
  plt.xlim(0, run)
  plt.ylim(0, 2)
  plt.ylabel('Mean # Heads')
  plt.xlabel('Number of runs')
  plt.title('Expectation[#Heads in Two Throws of a Fair Dice]')
  graph = plt.scatter([], [])

  def animate(i):
    graph.set_offsets(np.vstack((x_np[:i+1], y_np[:i+1])).T)
    graph.set_sizes(sizeI[:i+1])
    graph.set_facecolors(colors[:i+1])
    return graph

  ani = FuncAnimation(fig, animate, repeat=True, interval=40)
  plt.show()

plotNumHeads(10000)