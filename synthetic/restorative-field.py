import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style
sns.set_style("white")

# Create a figure
plt.figure(figsize=(10, 10))

# Style the plot
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

# Create grid points
x = np.linspace(-4, 4, 15)
y = np.linspace(-4, 4, 15)
X, Y = np.meshgrid(x, y)

# Calculate radius at each point
R = np.clip(np.sqrt(X**2 + Y**2), 0.01, None)

# Set the angle phi for the spiral (in radians)
phi = np.pi/2-np.pi/12

h = 2

# Calculate field components - inward field with rotation
# U and V components combine radial and angular components
U = np.where(R > h, (-X/R * np.cos(phi) - Y/R * np.sin(phi)), 0)
V = np.where(R > h, (-Y/R * np.cos(phi) + X/R * np.sin(phi)), 0)

idx_nonzero = (np.abs(U) > 0) & (np.abs(V) > 0)
X = X[idx_nonzero]
Y = Y[idx_nonzero]
U = U[idx_nonzero]
V = V[idx_nonzero]

# Plot the vector field
plt.quiver(X, Y, U, V, scale=20, width=0.0027, color='#EBB901', 
          headaxislength=5, headwidth=4, pivot='tip')

# Plot horizon circle and quivers
theta = np.linspace(0, 2*np.pi, 100)
plt.plot(h*np.cos(theta), h*np.sin(theta), 'k', linewidth=1.5, linestyle='dotted')

plt.xlim(-3.5, 3.5)
plt.ylim(-3.5, 3.5)
plt.gca().set_aspect('equal')

# Remove spines and ticks
sns.despine(top=True, right=True, left=True, bottom=True)
plt.gca().set_xticks([])
plt.gca().set_yticks([])

# Add text label for horizon
plt.text(h-1.05, 0.23, 'h', fontsize=18, color="gray")
plt.text(0.1, -0.3, r'$0$', fontsize=18)
plt.plot([0,h*np.cos(np.pi/6)],[0,h*np.sin(np.pi/6)], color='gray', linestyle='dotted')

# Add origin marker and label
plt.plot(0, 0, 'ko', markersize=4)
plt.text(0.1, -0.3, r'$0$', fontsize=18)

plt.savefig('figures/restorative-field.pdf', bbox_inches='tight')
plt.close()