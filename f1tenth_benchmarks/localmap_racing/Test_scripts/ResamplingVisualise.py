import numpy as np
import matplotlib.pyplot as plt

# Number of particles (for visualization purposes)
NP = 100

# Initialize tunable particle values (2D coordinates for simplicity)
# You can manually adjust these values to see how they affect resampling
x_values = np.linspace(0, 1, NP)
y_values = np.sin(2 * np.pi * x_values) * 0.5 + 0.5  # Example: sine wave distribution

particles = np.column_stack((x_values, y_values))

# Initialize weights with a base value
base_weight = 0.01
weights = np.full(NP, base_weight)

# Manually adjust weights for specific particles
weights[10:20] = 0.05  # Slightly increase weights for particles 10 to 20
weights[30:40] = 0.1   # Increase weights for particles 30 to 40
weights[60:90] = 0.2   # Significantly increase weights for particles 60 to 70

# Normalize the weights
weights /= np.sum(weights)

# Particle indices
particle_indices = np.arange(NP)

# Resampling based on weights
proposal_indices = np.random.choice(particle_indices, NP, p=weights)
proposal_distribution = particles[proposal_indices, :]

# Visualization
plt.figure(figsize=(10, 6))

# Plot original particles
plt.scatter(particles[:, 0], particles[:, 1], color='blue', alpha=0.5, label='Original Particles')

# Highlight resampled particles (proposal distribution)
plt.scatter(proposal_distribution[:, 0], proposal_distribution[:, 1], color='red', alpha=0.7, label='Resampled Particles')

# Optionally, draw lines showing resampling (from original to proposal)
for i in range(NP):
    plt.plot([particles[proposal_indices[i], 0], proposal_distribution[i, 0]],
             [particles[proposal_indices[i], 1], proposal_distribution[i, 1]], color='gray', alpha=0.3)

plt.title('Particle Resampling Visualization')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
