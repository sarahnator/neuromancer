import numpy as np
import matplotlib.pyplot as plt

def diffusion_limited_aggregation(grid_size=301, n_particles=1000):
    grid = np.zeros((grid_size, grid_size), dtype=bool)

    # Start with a single seed in the center
    center = grid_size // 2
    grid[center, center] = True

    # Directions for random walk (N, S, E, W)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def random_walk():
        # Start far from the center
        r = int(0.9 * center)
        theta = 2 * np.pi * np.random.rand()
        x = int(center + r * np.cos(theta))
        y = int(center + r * np.sin(theta))
        return x, y

    for _ in range(n_particles):
        x, y = random_walk()
        while True:
            # Perform random walk
            dx, dy = directions[np.random.randint(0, 4)]
            x = (x + dx) % grid_size
            y = (y + dy) % grid_size

            # Check if adjacent to cluster
            for nx, ny in directions:
                if grid[(x + nx) % grid_size, (y + ny) % grid_size]:
                    grid[x, y] = True
                    break
            if grid[x, y]:
                break

    return grid

# Generate DLA pattern
dla_grid = diffusion_limited_aggregation(grid_size=201, n_particles=70)

# Plot the result
plt.figure(figsize=(8, 8))
plt.imshow(dla_grid, cmap='inferno', origin='lower')
plt.title("Diffusion-Limited Aggregation (Lightning-like Pattern)")
plt.axis('off')
plt.show()
