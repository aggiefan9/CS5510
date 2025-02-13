import numpy as np
import matplotlib.pyplot as plt

# Constants
length = 35  # length of the robot (in feet)
width = 10   # width of the robot (in feet)
velocity = 8  # constant velocity (in m/s)
frequency = 2  # frequency of commands (in Hz)
radius = 18  # radius of the circle (in meters)

# Conversion constants
feet_to_meters = 0.3048

# Robot dimensions in meters
length_m = length * feet_to_meters
width_m = width * feet_to_meters

# Calculate time step
dt = 1 / frequency

# Calculate number of steps
num_steps = int(2 * np.pi * radius / velocity / dt)

omega = velocity / radius
path_to_edge = int((np.pi* dt) / omega)+2
# Initialize arrays to store trajectory and velocities
trajectory = np.zeros((int(num_steps+path_to_edge+1), 2))
# angular_velocities = np.zeros((int(num_steps+omega+1),))

for i in range (path_to_edge):
    t = i * dt
    # angular_velocities[i] = omega
    trajectory[i, 0] = 0  # Center of the circle in x-direction
    trajectory[i, 1] = velocity * t  # Move along the radius in y-direction

# Ackermann model simulation
for i in range(num_steps-1):
    t = i * dt
    # angular_velocities[i] = omega

    # Update x, y coordinates
    trajectory[i+path_to_edge, 0] = radius * np.sin(omega * t)
    trajectory[i+path_to_edge, 1] = radius - radius * np.cos(omega * t)

# Plot the resulting path
plt.figure(figsize=(10, 5))
plt.plot(trajectory[:, 0], trajectory[:, 1], label='Path')
plt.title('Ackermann Model - Path')
plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')
plt.legend()
plt.grid(True)
plt.show()

# Plot the trajectory and angular velocities
plt.figure(figsize=(10, 5))
plt.plot(trajectory[:, 0], label='X')
plt.plot(trajectory[:, 1], label='Y')
# plt.plot(angular_velocities, label='Angular Velocity')
plt.title('Ackermann Model - Trajectory and Angular Velocities')
plt.xlabel('Time Steps')
plt.legend()
plt.grid(True)
plt.show()
