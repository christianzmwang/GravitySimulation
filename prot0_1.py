


import numpy as np
import matplotlib.pyplot as plt

"""
Create Your Own N-body Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz
Simulate orbits of stars interacting due to gravity
Code calculates pairwise forces according to Newton's Law of Gravity

Edited to 2D
"""


def getAcc(pos, mass, G, softening):
    """
Calculate the acceleration on each particle due to Newton's Law 
    pos  is an N x 3 matrix of positions
    mass is an N x 1 vector of masses
    G is Newton's Gravitational constant
    softening is the softening length
    a is N x 3 matrix of accelerations
    """
    # positions r = [x,y,z] for all particles
    x = pos[:, 0:1]
    y = pos[:, 1:2]

    # matrix that stores all pairwise particle separations: r_j - r_i
    dx = x.T - x
    dy = y.T - y

    # matrix that stores 1/r^3 for all particle pairwise particle separations
    inv_r3 = (dx**2 + dy**2 + softening**2)
    inv_r3[inv_r3 > 0] = inv_r3[inv_r3 > 0]**(-1.5)

    ax = G * (dx * inv_r3) @ mass
    ay = G * (dy * inv_r3) @ mass

    # pack together the acceleration components
    a = np.hstack((ax, ay))
    print(a)
    return a

def main():
    """ N-body simulation """

    # Simulation parameters
    N = 2    # Number of particles
    t = 0      # current time of the simulation
    tEnd = 10.0   # time at which simulation ends
    dt = 0.01   # timestep
    softening = 0.1    # softening length
    G = 1.0    # Newton's Gravitational Constant
    plotRealTime = True  # switch on for plotting as the simulation goes along

    # Generate Initial Conditions
    np.random.seed(17)  # set the random number generator seed

    #Star, Planet, Planet, Meteor
    mass = np.array([[36], [0.1]])  #20.0*np.ones((N, 1))/N   #total mass of particles is 20
    pos = np.array([[0.0, 0.0], [2.0, -3.0]])  # np.random.randn(N,3)   # randomly selected positions and velocities
    vel = np.array([[0.0, 0.0], [0.0, 3.0]])  # np.random.randn(N,3)

    # Convert to Center-of-Mass frame
    vel -= np.mean(mass * vel, 0) / np.mean(mass)

    # calculate initial gravitational accelerations
    acc = getAcc(pos, mass, G, softening)

    # number of timesteps
    Nt = int(np.ceil(tEnd/dt))

    # save energies, particle orbits for plotting trails
    pos_save = np.zeros((N, 2, Nt+1))
    pos_save[:, :, 0] = pos
    t_all = np.arange(Nt+1)*dt

    # prep figure
    fig = plt.figure(figsize=(4, 5), dpi=80)
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0:2, 0])
    ax2 = plt.subplot(grid[2, 0])

    # Simulation Main Loop
    for i in range(Nt):
        # (1/2) kick
        vel += acc * dt/2.0

        # drift
        pos += vel * dt

        # update accelerations
        acc = getAcc(pos, mass, G, softening)

        # (1/2) kick
        vel += acc * dt/2.0

        # update time
        t += dt

        # save energies, positions for plotting trail
        pos_save[:, :, i+1] = pos

        # plot in real time
        if plotRealTime or (i == Nt-1):
            plt.sca(ax1)
            plt.cla()
            xx = pos_save[:, 0, max(i-50, 0):i+1]
            yy = pos_save[:, 1, max(i-50, 0):i+1]
            plt.scatter(xx, yy, s=1, color=[.7, .7, 1])
            plt.scatter(pos[:, 0], pos[:, 1], s=10, color='blue')
            ax1.set(xlim=(-2, 2), ylim=(-2, 2))
            ax1.set_aspect('equal', 'box')
            ax1.set_xticks([i for i in range(-3, 6)])
            ax1.set_yticks([i for i in range(-3, 4)])

            plt.sca(ax2)
            plt.cla()
            ax2.set(xlim=(0, tEnd), ylim=(-1000, 1000))
            ax2.set_aspect(0.0006)

            plt.pause(0.0001)

        if pos[1, 0] <= 0:
          break 

    print("hey")

    # add labels/legend
    plt.sca(ax2)
    plt.xlabel('time')
    plt.ylabel('energy')
    ax2.legend(loc='upper right')

    # Save figure
    plt.savefig('nbody.png', dpi=240)
    plt.show()

    return 0


if __name__ == "__main__":
    main()



