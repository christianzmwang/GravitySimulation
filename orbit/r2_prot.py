


import numpy as np
import matplotlib.pyplot as plt

def getAcc(pos, mass, G, softening):

    # x and y position of both particles 
    x = pos[:, 0:1]
    y = pos[:, 1:2]

    dx = x[0] - x[1]
    dy = y[0] - y[1]


    # matrix that stores 1/r^2 for all particle pairwise particle separations
    inv_r2 = (dx**2 + dy**2 + softening) ** -1

    ax = float(G * (dx * inv_r2) * (mass[0] * mass[1]) )
    ay = float(G * (dy * inv_r2) * (mass[0] * mass[1]) )
        
    return np.array([ax, ay])


def main():

    # Simulation parameters
    N = 2    # Number of particles
    t = 0      # current time of the simulation
    tEnd = 10.0   # time at which simulation ends
    dt = 0.05   # timestep
    G = 1.0    # Newton's Gravitational Constant
    softening = 0.2
    plotRealTime = True  # switch on for plotting as the simulation goes along

    # Star, Planet
    mass = np.array([[100], [1]])  
    pos = np.array([[0.0, 0.0], [2.0, 0.0]])  
    vel = np.array([0.0, 10.0])  

    # calculate initial gravitational accelerations
    acc = getAcc(pos, mass, G, softening)

    # number of timesteps
    Nt = int(np.ceil(tEnd/dt))

    # save energies, particle orbits for plotting trails
    pos_save = np.zeros((N, 2, Nt+1))
    pos_save[:, :, 0] = pos
    t_all = np.arange(Nt+1)*dt

    # prep figure
    fig = plt.figure(figsize=(8, 8), dpi=80)
    grid = plt.GridSpec(1, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[:, 0])


    for i in range(Nt):
        # (1/2) kick
        vel += acc * dt/2.0

        # drift
        pos[1] += vel * dt

        # update accelerations
        acc = getAcc(pos, mass, G, softening)

        # (1/2) kick
        vel += acc * dt/2.0

        # update time
        t += dt

        # save positions for plotting trail
        pos_save[:, :, i+1] = pos

        # plot in real time
        if plotRealTime or (i == Nt-1):
            plt.sca(ax1)
            plt.cla()
            xx = pos_save[:, 0, 0:i+1]
            yy = pos_save[:, 1, 0:i+1]
            plt.scatter(xx, yy, s=1, color=[.7, .7, 1])
            plt.scatter(pos[:, 0], pos[:, 1], s=10, color='blue')
            ax1.set(xlim=(-2, 2), ylim=(-2, 2))
            ax1.set_aspect('equal', 'box')
            ax1.set_xticks([i for i in range(-10, 11)])
            ax1.set_yticks([i for i in range(-10, 11)])


            plt.pause(0.0001)

    # Save figure
    plt.savefig('nbody.png', dpi=240)
    plt.show()

    return 0


if __name__ == "__main__":
    main()