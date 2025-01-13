import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style
sns.set_style("white")

# Create a figure
plt.figure(figsize=(10, 10))

# Create grid points
x = np.linspace(-4, 4, 20)
y = np.linspace(-4, 4, 20)
X, Y = np.meshgrid(x, y)

# Calculate radius at each point
R = np.sqrt(X**2 + Y**2)

# Define horizon
h = 2

# Calculate field components
# Inside horizon: zero field
# Outside horizon: point inward with magnitude 1/r
U = np.where(R > h, -X/R, 0)
V = np.where(R > h, -Y/R, 0)

# Plot the field lines
idx_nonzero = (np.abs(U) > 0) & (np.abs(V) > 0)
X = X[idx_nonzero]
Y = Y[idx_nonzero]
U = U[idx_nonzero]
V = V[idx_nonzero]

# Set axis properties
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)


# Plot horizon circle and quivers
theta = np.linspace(0, 2*np.pi, 97)
theta_arrows = theta[::3]
theta2_arrows = theta[::2]
plt.quiver(h*np.cos(theta_arrows), h*np.sin(theta_arrows), -np.cos(theta_arrows), -np.sin(theta_arrows), scale=20, width=0.0027, color='#EBB901', headaxislength=5, headwidth=4, pivot='tip')
plt.quiver(1.3*h*np.cos(theta2_arrows), 1.3*h*np.sin(theta2_arrows), -np.cos(theta2_arrows), -np.sin(theta2_arrows), scale=20, width=0.0027, color='#EBB901', headaxislength=5, headwidth=4, pivot='tip')
plt.plot(h*np.cos(theta), h*np.sin(theta), 'k', linewidth=1.5, linestyle='dotted')



# Set limits and aspect ratio
plt.xlim(-3.5, 3.5)
plt.ylim(-3.5, 3.5)
plt.gca().set_aspect('equal')

# Style the axes
sns.despine(top=True, right=True, left=True, bottom=True)
plt.gca().set_xticks([])
plt.gca().set_xticklabels([])
plt.gca().set_yticks([])
plt.gca().set_yticklabels([])

# Add text label for horizon
plt.text(h-1.05, 0.23, 'h', fontsize=18, color="gray")
plt.text(0.1, -0.3, r'$0$', fontsize=18)
plt.plot([0,h*np.cos(np.pi/6)],[0,h*np.sin(np.pi/6)], color='gray', linestyle='dotted')

# Plot marker at origin
plt.plot(0, 0, 'ko', markersize=4)

plt.savefig('figures/restorative-field.pdf', bbox_inches='tight')