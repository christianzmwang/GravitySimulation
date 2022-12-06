

import numpy as np
import matplotlib.pyplot as plt
import os

#Creates a folder named "figures" to store figures
img_dir = os.path.join(os.getcwd(), r"figures")
if not os.path.exists(img_dir):
  os.makedirs(img_dir)

#Get acceleration function
def getAcc(pos, mass, G, softening):

  # Positions r = [x, y]
  x = pos[:, 0:1]
  y = pos[:, 1:2]

  # Matrix that stores all pairwise particle seperations
  dx = x.T - x
  dy  = y.T - y

  # matrix that stores 1/r^2 for all particle pairwise particle separations
  inv_r2 = (dx**2 + dy**2 + softening**2) ** 0.5
  inv_r2[inv_r2 > 0] = inv_r2[inv_r2 > 0]**(-1.5)

  ax = G * (dx * inv_r2) @ mass
  ay = G * (dy * inv_r2) @ mass

  # pack together the acceleration components
  a = np.hstack((ax, ay))
  return a

def main(s):
    """ N-body simulation """

    # Simulation parameters
    softening = 0.1    # softening length
    G = 1.0    # Newton's Gravitational Constant
    plotRealTime = True  # switch on for plotting as the simulation goes along
  
    #General initial conditions 
    mass = np.array([[s], [1]])  
    pos = np.array([[0.0, 0.0], [-1000000.0, 0.0]])  
    vel = np.array([[0.0, 0.0], [0.0, 0.0]])  

    # Convert to Center-of-Mass frame
    vel -= np.mean(mass * vel, 0) / np.mean(mass)

    # calculate initial gravitational accelerations
    acc = getAcc(pos, mass, G, softening)

    # save particle orbits for plotting trails
    pos_save = np.zeros((2, 2, 10001))
    pos_save[:, :, 0] = pos

    #Save acceleration and velocity
    acc_vel_save = [round(acc[1, 0]), vel[1, 0]]

    # prep figure
    plt.figure(figsize=(6, 6), dpi=80)

    # Simulation Main Loop
    for i in range(10000):
        # (1/2) kick
        vel += acc/2.0

        # drift
        pos[1] += vel[1]

        # update accelerations
        acc = getAcc(pos, mass, G, softening)
        print(acc)

        # (1/2) kick
        vel += acc/2.0

        # save positions for plotting trail
        pos_save[:, :, i+1] = pos
 
        # save acceleration and velocity
        acc_vel_save.append( [round(acc[1, 0]), round(vel[1, 0])] )

        # plot in real time
        if plotRealTime:
            plt.plot()
            plt.cla()
            xx = pos_save[:, 0, max(i-50, 0):i+1]
            yy = pos_save[:, 1, max(i-50, 0):i+1]
            plt.scatter(xx, yy, s=1, color=[.7, .7, 1])
            plt.scatter(pos[:, 0], pos[:, 1], s=10, color='blue')

            plt.pause(0.0001)
        
        # end simulation when projectile hits stationary body
        if pos[1, 0] >= 0:
          print(i)
          break

    # Save figure
    plt.savefig('figures/nbody{}.png'.format(s), dpi=240)
    plt.show()
    print(acc_vel_save)

    return 0


main(100)





