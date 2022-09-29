

import numpy as np
import matplotlib.pyplot as plt

def getAcc(pos, mass, G, softening):
  
  # Positions r = [x, y]
  x = pos[:, 0:1]
  y = pos[:, 1:2]

  # Matrix that stores all pairwise particle seperations
  dx = x.T - x
  dy  = y.T - y


