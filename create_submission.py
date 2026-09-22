import os
import zipfile

# Kept at their folder paths in the zip: ex1, ex2 and ex3 each have a robot.py,
# and flattening would leave only one of them.
CODE = [
    os.path.join("ex1", "robot.py"),
    os.path.join("ex1", "sample_velocities.py"),
    os.path.join("ex1", "jacobian_example.py"),
    os.path.join("ex2", "robot.py"),
    os.path.join("ex2", "IK_example.py"),
    os.path.join("ex3", "robot.py"),
    os.path.join("ex3", "linear_joint_trajectory.py"),
    os.path.join("ex3", "linear_workspace_trajectory.py"),
    os.path.join("ex4", "grav_comp.py"),
]


def create_submission():
    missing = [path for path in CODE if not os.path.exists(path)]
    if missing:
        raise SystemExit(
            "Missing " + ", ".join(missing) + " — run this from the assignment root."
        )

    with zipfile.ZipFile("handin.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for path in CODE:
            zf.write(path, path)

    print("Created submission archive: handin.zip")
    for path in CODE:
        print(f"  {path}")


if __name__ == "__main__":
    create_submission()
