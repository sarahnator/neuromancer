import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
L = 1.0        # Length of string
T = 2.0        # Total time
nx = 100       # Spatial points
nt = 500       # Time steps
c = 1.0        # Wave speed
dx = L / (nx - 1)
dt = T / nt
r = c * dt / dx

# Grid
x = np.linspace(0, L, nx)
u = np.zeros((nt, nx))

# Initial condition: pluck the string
u[0, :] = np.sin(np.pi * x)
u[1, 1:-1] = u[0, 1:-1] + 0.5 * r**2 * (u[0, 2:] - 2 * u[0, 1:-1] + u[0, :-2])

# Time stepping
for n in range(1, nt - 1):
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
plt.show()
