import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
L = 1.0        # Length of string
T = 2.0        # Total time
nx = 100       # Spatial points
nt = 500       # Time steps
c = 1.0        # Wave speed
dx = L / (nx - 1) # Fix the number of points to represent the full length of the string
dt = T / nt        # Fix the number of time steps to represent the full time
r = c * dt / dx

# Grid: use the Eulerian approach (fix a grid in space observe how quantities evolve in time at those grid points)
x = np.linspace(0, L, nx)
u = np.zeros((nt, nx))

# Initial condition: pluck the string
u[0, :] = np.sin(np.pi * x)
# Compute the first time step separately from the loop because we need two time steps to compute the next one
# and we only have one time step to compute the first one
# use a second-order taylor expansion to compute the first time step (initial velocity = 0)
u[1, 1:-1] = u[0, 1:-1] + 0.5 * r**2 * (u[0, 2:] - 2 * u[0, 1:-1] + u[0, :-2])

"""
The wave equation is a second order PDE. We can use finite difference methods to approximate the solution.
The finite difference method is a numerical method for solving PDEs by approximating the derivatives with finite differences.
The wave equation is given by:
    u_tt = c^2 * u_xx
    where u_tt is the second derivative of u with respect to time and u_xx is the second derivative of u with respect to space.
    We can use finite difference methods to approximate the second derivative in space and time.
    The finite difference approximation for the second derivative in space is given by:
    
    u_xx = (u[i+1] - 2*u[i] + u[i-1]) / dx^2
    The finite difference approximation for the second derivative in time is given by:
    
    u_tt = (u[n+1] - 2*u[n] + u[n-1]) / dt^2

    We can rearrange the wave equation to get:
    u[n+1] = 2*(1 - r^2)*u[n] - u[n-1] + r^2*(u[n][i+1] + u[n][i-1]) --> our update rule
    where r = c*dt/dx is the Courant number.
    The Courant number is a dimensionless number that measures the ratio of the wave speed to the grid speed.
    The Courant number must be less than 1 for the finite difference method to be stable.
"""
# Time stepping
for n in range(1, nt - 1): # Note the dirichlet boundary conditions u[0] = 0 and u[L] = 0 for all time so they are not updated
    # Update the interior points using the finite difference method, since by the dirichlet boundary the wave only evolves on the interior points
    u[n + 1, 1:-1] = (
        2 * (1 - r**2) * u[n, 1:-1]
        - u[n - 1, 1:-1]
        + r**2 * (u[n, 2:] + u[n, :-2])
    )

# Animation
fig, ax = plt.subplots()
line, = ax.plot(x, u[0])
ax.set_ylim(-1.1, 1.1)
ax.set_title("1D Wave Equation")

def update(frame):
    line.set_ydata(u[frame])
    ax.set_title(f"Time step: {frame}")
    return line,

ani = animation.FuncAnimation(fig, update, frames=nt, interval=20, blit=True)
# plt.show()
ani.save("wave.gif", writer='pillow', fps=30)
