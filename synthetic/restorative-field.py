import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style
sns.set_style("white")

# Create a figure
plt.figure(figsize=(10, 10))

# Create grid points
B = 4
x = np.linspace(-B, B, 20)
y = np.linspace(-B, B, 20)
X, Y = np.meshgrid(x, y)

# Calculate radius at each point
R = np.sqrt(X**2 + Y**2)

# Define horizon
h = 2.15

# Calculate field components
# Inside horizon: zero field
# Outside horizon: point inward with magnitude 1/r
U = np.where(R > h, -X/R, 0)
V = np.where(R > h, -Y/R, 0)

scale = 20
Rho = np.sqrt(U**2 + V**2)
Rho = np.where(Rho > 0, Rho, 1)
XplusU = X + U / Rho * 2*B/scale
YplusV = Y + V / Rho * 2*B/scale
T = np.sqrt(XplusU**2 + YplusV**2) # Radius of arrowhead
Numer = np.where((R > h) & (T < h), R - h, 1)
Denom = np.where((R > h) & (T < h), R - T, 1)
Factor = Numer / Denom
U = U / Rho * 2*B/scale * Factor
V = V / Rho * 2*B/scale * Factor

# Plot the field lines
idx_nonzero = (np.abs(U) > 0) & (np.abs(V) > 0)
X = X[idx_nonzero]
Y = Y[idx_nonzero]
U = U[idx_nonzero]
V = V[idx_nonzero]

# Set axis properties
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

# Plot quivers
plt.quiver(X, Y, U, V, scale=1, scale_units='xy', angles='xy', width=0.0027, color='#EBB901', headaxislength=5, headwidth=4, pivot='tail')

# Plot horizon circle
h = 2
theta = np.linspace(0, h*np.pi, 97)
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
