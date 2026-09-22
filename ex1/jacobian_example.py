import numpy as np
from robot import Robot


def compute_joint_torques(robot, thetas, wrench):
    """
    Returns the joint torques needed to apply `wrench` with the end effector.

    Parameters:
    robot : Robot object
    thetas : numpy array of shape (dof,), the joint angles
    wrench : numpy array of shape (3,), [F_x, F_y, M]

    Returns:
    Tau : numpy array of shape (dof,)
    """
    # --------------- BEGIN STUDENT SECTION (Compute Joint Torques) ---------------
    # TODO: Compute the Jacobian at the end effector and calculate the joint torques

    # Hints:
    # - Use the jacobians method to compute the Jacobian matrix.
    # - Compute Tau as the transpose of the Jacobian times the wrench.

    # Your code starts here

    Tau = None

    # Your code ends here
    # --------------- END STUDENT SECTION ------------------------------------------

    return Tau


if __name__ == "__main__":
    # Create the 10R robot
    link_lengths_10R = np.ones(10) * 5
    link_masses_10R = np.ones(10)
    joint_masses_10R = np.ones(10)
    robot10R = Robot(
        link_lengths=link_lengths_10R,
        link_masses=link_masses_10R,
        joint_masses=joint_masses_10R,
        end_effector_mass=1
    )

    # Define the joint angles
    thetas_10R = np.array([0, 0.1, 0.2, 0.3, 0.4, 0, -0.1, -0.2, -0.3, -0.4])

    # The wrench to apply: forces in x and y, then a moment
    wrench = np.array([0, 5, 2])

    Tau = compute_joint_torques(robot10R, thetas_10R, wrench)

    # Print the resulting joint torques
    print("\nJoint Torques to apply the given wrench at the end effector:")
    print(Tau)
