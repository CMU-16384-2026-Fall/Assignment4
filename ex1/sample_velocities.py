# sample_velocities.py

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from robot import Robot

HERE = Path(__file__).resolve().parent


def compute_kinematics(robot, theta, theta_dot):
    """
    Returns the end effector position and velocity at every time step.

    Parameters:
    robot : Robot object
    theta : numpy array of shape (n, dof), joint angles; row i is time step i
    theta_dot : numpy array of shape (n, dof), joint velocities

    Returns:
    x, y, vx, vy : numpy arrays of shape (n,)
    """
    # Initialize arrays
    n = len(theta)
    x = np.zeros(n)
    y = np.zeros(n)
    vx = np.zeros(n)
    vy = np.zeros(n)

    # --------------- BEGIN STUDENT SECTION (Compute Positions and Velocities) ---------------
    # TODO: Calculate positions and velocities for each timestep

    # Hints:
    # - Loop through each time step.
    # - Use the robot's forward kinematics to compute positions.
    # - Use the jacobians method to compute velocities.

    # Your code starts here

    ...

    # Your code ends here
    # --------------- END STUDENT SECTION ----------------------------------------------------

    return x, y, vx, vy


def sample_velocities():
    # Load sample data
    data = np.load(HERE / 'sample_ground_truth.npz')
    time = data['time'].flatten()
    theta = data['theta']
    theta_dot = data['theta_dot']
    ground_truth_x = data['ground_truth_x'].flatten()
    ground_truth_y = data['ground_truth_y'].flatten()
    ground_truth_vx = data['ground_truth_vx'].flatten()
    ground_truth_vy = data['ground_truth_vy'].flatten()

    # Create robot (matching sample data)
    robot = Robot(
        link_lengths=np.array([1.5, 2.5]),
        link_masses=np.array([1.5, 2.5]),
        joint_masses=np.array([1, 1]),
        end_effector_mass=1
    )

    x, y, vx, vy = compute_kinematics(robot, theta, theta_dot)

    # Sub-sample the data for plotting
    subsample_resolution = 10
    x_sub = x[::subsample_resolution]
    y_sub = y[::subsample_resolution]
    vx_sub = vx[::subsample_resolution]
    vy_sub = vy[::subsample_resolution]

    # Plot path data and ground truth
    plt.figure()
    plt.plot(x, y, 'k-', linewidth=1, label='Your Kinematics')
    plt.plot(ground_truth_x, ground_truth_y, 'g--', linewidth=1, label='Correct Kinematics')
    plt.quiver(x_sub, y_sub, vx_sub, vy_sub, angles='xy', scale_units='xy', scale=0.5, color='black')
    plt.title('Plot of end effector position and velocity over a sample run.')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend(loc='lower center')
    plt.axis('equal')
    plt.show()

    # Plot velocities
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(time, vx, 'k-', linewidth=1, label='Your Velocity')
    plt.plot(time, ground_truth_vx, 'g--', linewidth=1, label='Correct Velocity')
    plt.xlabel('t')
    plt.ylabel('v_x [m/s]')
    plt.legend(loc='lower right')

    plt.subplot(2, 1, 2)
    plt.plot(time, vy, 'k-', linewidth=1, label='Your Velocity')
    plt.plot(time, ground_truth_vy, 'g--', linewidth=1, label='Correct Velocity')
    plt.xlabel('t')
    plt.ylabel('v_y [m/s]')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    sample_velocities()
