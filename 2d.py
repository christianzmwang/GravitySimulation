
import numpy as np
import matplotlib.pyplot as plt
import os

#Creates a folder named "figures" to store figures
img_dir = os.path.join(os.getcwd(), r"figures")
if not os.path.exists(img_dir):
  os.makedirs(img_dir)
d = [ [], [] ]
#Get acceleration function
def getAcc(pos, mass, G, softening):

  # Positions r = [x, y]
  x = pos[:, 0:1]
  y = pos[:, 1:2]

  # Matrix that stores all pairwise particle seperations
  dx = x.T - x
  dy  = y.T - y

  # matrix that stores 1/r^2 for all particle pairwise particle separations
  inv_r2 = (dx**2 + dy**2 + softening**2)
  inv_r2[inv_r2 > 0] = inv_r2[inv_r2 > 0]**(-0.5)

  ax = G * (dx * inv_r2) @ mass
  ay = G * (dy * inv_r2) @ mass

  # pack together the acceleration components
  a = np.hstack((ax, ay))
  return a

def main(s):
    """ N-body simulation """

    # Simulation parameters
    N = 2    # Number of particles
    t = 0      # current time of the simulation
    tEnd = 10.0   # time at which simulation ends
    dt = 0.01   # timestep
    softening = 0.1    # softening length
    G = 1.0    # Newton's Gravitational Constant
    plotRealTime = 0  # switch on for plotting as the simulation goes along

    # Generate Initial Conditions
  
    #erere
    mass = np.array([[s], [1]])  
    pos = np.array([[0.0, 0.0], [50.0, -100.0]])  
    vel = np.array([[0.0, 0.0], [0.0, 500]])  
    dis = [ [( (pos[1][0] - pos[0][0])**2 + (pos[1][1] - pos[0][1])**2 )**0.5, list(pos[1]) ] ]

    # Convert to Center-of-Mass frame
    vel -= np.mean(mass * vel, 0) / np.mean(mass)

    # calculate initial gravitational accelerations
    acc = getAcc(pos, mass, G, softening)

    # number of timesteps
    Nt = int(np.ceil(tEnd/dt))

    # save particle orbits for plotting trails
    pos_save = np.zeros((N, 2, Nt+1))
    pos_save[:, :, 0] = pos

    # prep figure
    fig = plt.figure(figsize=(6, 6), dpi=80)
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0:2, 0])

    # Simulation Main Loop
    for i in range(Nt):
        # (1/2) kick
        vel += acc * dt/2.0

        # drift
        pos[1] += vel[1] * dt

        # store distance 
        dis.append( [(pos[1][0]**2 + pos[1][1]**2 )**0.5, list(pos[1]) ] )

        # update accelerations
        acc = getAcc(pos, mass, G, softening)

        # (1/2) kick
        vel += acc * dt/2.0

        # update time
        t += dt

        # save positions for plotting trail
        pos_save[:, :, i+1] = pos

        if pos[1, 0] <= 0:
          plt.sca(ax1)
          plt.cla()
          xx = pos_save[:, 0, :]
          yy = pos_save[:, 1, :]
          plt.scatter(xx, yy, s=1, color=[.7, .7, 1])
          plt.scatter(pos[:, 0], pos[:, 1], s=10, color='blue')
          ax1.set(xlim=(-100, 100), ylim=(-100, 100))
          ax1.set_aspect('equal', 'box')
          break 

    cInd = dis.index(sorted(dis)[0] )
    c = dis[cInd]
    d[0].append(c[0])
    d[1].append(s)
    plt.plot([0, c[1][0]], [0, c[1][1]], linestyle="--", color=[.7, .7, 1])
    plt.text(-20, -20, round(c[0], 2))

    # Save figure
    plt.savefig('figures/nbody{}.png'.format(s), dpi=240)
    #plt.show()
    plt.close()
    return 0

ss = list(range(10, 10000, 25))

for i in ss:
  main(i)

plt.plot(d[1], d[0])
plt.savefig("figures/test.png")
