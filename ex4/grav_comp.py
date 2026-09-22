import math
import time

import numpy as np

# Constants
g = 9.81  # gravitational acceleration (m/s^2)

# The xArm7 as a 2D RR arm in the vertical x-z plane (x forward, z up).
#
# With joints 1, 3, 5 and 7 at zero, the elbow (joint 4) and the wrist (joint 6)
# turn about axes parallel to y, so everything they carry moves in the x-z plane:
#
#   link A: the forearm, from the elbow (joint 4) to the wrist (joint 6).
#           Its absolute angle theta_a is measured from +x toward +z.
#   link B: everything past the wrist (links 6, 7 and the gripper), pointing
#           from joint 6 toward the flange. Its angle theta_b is measured
#           relative to link A, also counter-clockwise.
#
# Each centre of mass is given in its own link's frame: the first entry is along
# the link, the second perpendicular to it (counter-clockwise). The COMs are
# not exactly on the link lines, and ignoring that costs up to ~1 N*m.
#
# All the numbers are measured from the robot model the simulator runs.

LINK_A_LENGTH = 0.351159                # m, elbow to wrist
COM_A = np.array([0.213927, 0.026826])  # m, in link A's frame, from the elbow
COM_B = np.array([0.105635, 0.005510])  # m, in link B's frame, from the wrist
MASS_A = 3.0472                         # kg, links 4 and 5
MASS_B = 2.246430                       # kg, links 6 and 7 and the gripper

# How the planar angles relate to the arm's joint values. Joint 4 turns about
# -y, joint 6 (and joint 2, the shoulder) about +y.
ELBOW_OFFSET = -1.348266  # rad, theta_a when joints 2 and 4 are at zero
WRIST_OFFSET = 0.442072   # rad, theta_b when joint 6 is at zero


def planar_angles(q):
    """(theta_a, theta_b) for the arm's seven joint values q. Do not modify."""
    return ELBOW_OFFSET + q[3] - q[1], WRIST_OFFSET - q[5]


def to_joint_torques(torques):
    """The arm's joint-4 and joint-6 torques for planar [tau_a, tau_b]. Do not modify."""
    return np.array([torques[0], -torques[1]])


# Function to compute the Jacobians of the two centres of mass
def compute_jacobians(theta_a, theta_b):
    """
    TODO: Implement the Jacobian of each link's centre of mass.

    Parameters:
    - theta_a (float): Absolute angle of link A (the forearm) in radians.
    - theta_b (float): Angle of link B relative to link A in radians.

    Returns:
    - J_A (np.ndarray): 2x2 Jacobian of link A's centre of mass. Rows are
      (x, z) and columns are (theta_a, theta_b).
    - J_B (np.ndarray): 2x2 Jacobian of link B's centre of mass.
    """
    # --------------- BEGIN STUDENT SECTION ----------------------------------
    # HINT: Write the position of each centre of mass as a function of
    # theta_a and theta_b (rotate COM_A / COM_B into the plane), then
    # differentiate.
    # Replace the following lines with your implementation
    raise NotImplementedError
    # --------------- END STUDENT SECTION ------------------------------------

    return J_A, J_B

# Function to compute the gravitational torques using the Jacobians
def compute_gravitational_torques(theta_a, theta_b):
    """
    TODO: Implement the computation of gravitational torques using the Jacobians.

    Parameters:
    - theta_a (float): Absolute angle of link A in radians.
    - theta_b (float): Angle of link B relative to link A in radians.

    Returns:
    - torques (np.ndarray): [tau_a, tau_b], the torques the two joints must
      supply to hold the arm still against gravity.
    """
    # --------------- BEGIN STUDENT SECTION ----------------------------------
    # HINT: A joint holding a mass m against gravity supplies J^T [0, m g].
    # Replace the following lines with your implementation
    raise NotImplementedError
    # --------------- END STUDENT SECTION ------------------------------------

    return torques


# Poses for the sweep below: the shoulder upright, the forearm level, and the
# wrist lowered 30 degrees at a time.
ELBOW_LEVEL = -ELBOW_OFFSET  # joint 4 value that makes theta_a = 0
TORQUE_TOLERANCE = 0.1  # N*m
SETTLE_TIME = 0.5  # s to let the arm come to rest before reading torques

if __name__ == "__main__":
    # The course's xArm7 library, imported here so that the functions above
    # can be tested without it. Always the simulator, even with ROBOT_IP set:
    # the real xArm7 has no joint torque sensors, so there is nothing on it to
    # check these torques against.
    from xarm7_lib import Robot

    arm = Robot(sim=True)
    joint_angles = np.array([0.0, 0.0, 0.0, ELBOW_LEVEL, 0.0, 0.0, 0.0])
    arm.set_joint_targets(joint_angles)
    initial_angle_6 = joint_angles[5]

    print("Gravity Compensation Verification\n")
    print("-" * 97)
    print(f"{'Joint 6 Angle (deg)':<22}{'Joint':<8}{'Calculated Torque (Nm)':<25}"
          f"{'Actual Torque (Nm)':<22}{'Difference (Nm)':<17}{'Result'}")
    print("-" * 97)

    try:
        # Increment joint 6 by 30 degrees, three times
        for i in range(3):
            joint_angles[5] = initial_angle_6 + math.radians(30 * (i + 1))
            arm.set_joint_targets(joint_angles)
            time.sleep(SETTLE_TIME)

            # Planar angles from where the arm actually is, not where it was sent
            theta_a, theta_b = planar_angles(arm.joint_values)
            calculated = to_joint_torques(compute_gravitational_torques(theta_a, theta_b))
            # What a joint torque sensor reads. Not joint_efforts, the motor
            # torque: at rest, joint friction carries part of the load, so
            # the motor can be off from gravity by up to ~1 N*m.
            actual = arm.joint_torques[[3, 5]]

            for joint, calc, act in zip((4, 6), calculated, actual):
                difference = abs(calc - act)
                result = "PASS" if difference <= TORQUE_TOLERANCE else "FAIL"
                print(f"{math.degrees(joint_angles[5]):<22.2f}{joint:<8}{calc:<25.3f}"
                      f"{act:<22.3f}{difference:<17.3f}{result}")
    finally:
        arm.stop()

    print("-" * 97)
    print("\nVerification Completed.")
