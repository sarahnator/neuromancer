import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

from MassSpringDamperExample import springmassdamper, rk4_delta_only

# Parameters
T = 5_000        # Total steps
dt = 0.05
zeta = 0.05
omega_n = 0.4
initial_conditions = torch.tensor([1.0, -0.4]).unsqueeze(0)
n_frames = 250

# Simulate
states = [initial_conditions]
for _ in range(T):
    current_state = states[-1]
    delta_X = rk4_delta_only(lambda x: springmassdamper(x, zeta, omega_n), current_state, dt)
    states.append(current_state + delta_X)
states = torch.cat(states, dim=0)  # Shape (T+1, 2)

# Downsample
downsample = T // n_frames
x_vals = states[::downsample, 0]
u_vals = states[::downsample, 1]
t = np.arange(len(x_vals)) * dt * downsample

# Plot setup
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
line1, = ax1.plot([], [], 'b-', label='Position')
line2, = ax2.plot([], [], 'r-', label='Velocity')
ax1.set_xlim(0, t[-1])
ax1.set_ylim(-1.5, 1.5)
ax2.set_xlim(0, t[-1])
ax2.set_ylim(-1.5, 1.5)
ax1.set_xlabel("Time (s)")
ax2.set_xlabel("Time (s)")
ax1.legend()
ax2.legend()

# Update function
def update(frame):
    line1.set_data(t[:frame+1], x_vals[:frame+1])
    line2.set_data(t[:frame+1], u_vals[:frame+1])
    current_time = float(t[frame])  
    ax1.set_title(f"Position (t={current_time:.2f}s)")
    ax2.set_title(f"Velocity (t={current_time:.2f}s)")
    return line1, line2

# Animate
ani = animation.FuncAnimation(fig, update, frames=len(t), interval=50, blit=False)
plt.tight_layout()
# Save the animation
# ani.save("mass_spring_damper.gif", writer='imagemagick', fps=30)
# plt.show()
ani.save("spring_damper.gif", writer='pillow', fps=20)