import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style
sns.set_style("white")

# Create a figure
plt.figure(figsize=(10, 10))

# Create grid points
x = np.linspace(-4, 4, 15)
y = np.linspace(-4, 4, 15)
X, Y = np.meshgrid(x, y)

# Calculate radius at each point
R = np.clip(np.sqrt(X**2 + Y**2), 0.01, None)

# Calculate field components - rotational field
U = np.where(R > 0.2, -Y/R, 0)
V = np.where(R > 0.2, X/R, 0)

# Plot the vector field
plt.quiver(X, Y, U, V, scale=20, width=0.0027, color='#CCCCCC', 
          headaxislength=5, headwidth=4, pivot='tip')

# Calculate trajectory
def get_field_at_point(x, y):
    r = max(np.sqrt(x**2 + y**2), 0.01)
    return -y/r, x/r  # Rotational field components

# Starting point
x0, y0 = 0.1, 0.1
points = [(x0, y0)]
x, y = x0, y0
learning_rate = 0.8

# Gradient descent
for _ in range(30):  # Increased steps to see more of the rotation
    dx, dy = get_field_at_point(x, y)
    x += learning_rate * dx
    y += learning_rate * dy
    points.append((x, y))
    
    # Stop if we reach the horizon
    if np.sqrt(x**2 + y**2) <= 0.2:
        break

# Convert points to arrays for plotting
trajectory = np.array(points)

# Plot trajectory
plt.plot(trajectory[:, 0], trajectory[:, 1], color='#EBB901', linewidth=2.5, linestyle='dotted', marker='o', markersize=9, alpha=0.7)
plt.plot(x0, y0, 'o', markeredgecolor='#EBB901', markerfacecolor='#EBB901', markersize=12)

# Style the plot
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
plt.xlim(-3.5, 3.5)
plt.ylim(-3.5, 3.5)
plt.gca().set_aspect('equal')

# Remove spines and ticks
sns.despine(top=True, right=True, left=True, bottom=True)
plt.gca().set_xticks([])
plt.gca().set_yticks([])

# Add origin marker and label
plt.plot(0, 0, 'ko', markersize=4)
plt.text(0.1, -0.3, r'$0$', fontsize=18)

plt.savefig('figures/expanding-trajectory.pdf', bbox_inches='tight')
plt.close()