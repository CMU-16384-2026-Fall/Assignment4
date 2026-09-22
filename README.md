# Assignment 4

Velocity kinematics, inverse kinematics, trajectories and gravity compensation.
The writeup is built from `hw4.tex`; the code is in `ex1/` to `ex4/`.

## Setup

Reinstall the course's robot library from the copy in this handout. It carries
the simulator and visualizer the demos use, and it may be newer than the one
you installed for an earlier assignment:

```bash
cd 16384-robot-lib
conda env update -f environment.yml    # or `conda env create -f environment.yml` on a new machine
conda activate 16384
cd ..
```

## Checking your work

```bash
python local_autograder.py            # every exercise
python local_autograder.py -p ex2     # one exercise: ex1, ex2, ex3 or ex4
python local_autograder.py -v         # list every failing case, not just the first few
```

This runs your code on the test cases in `exN/inK.txt` and compares the results
with `exN/outK.txt`. Gradescope runs the same tests on randomly perturbed
versions of these cases, with the answers computed fresh each time. Code that
is correct in general scores the same on both; code that only works for these
exact numbers does not.

One difference: `sample_velocities.py`, `jacobian_example.py` and
`linear_workspace_trajectory.py` are checked here using your own `robot.py`, so
they can only pass once your `Robot` does. Gradescope gives them a correct
`Robot` instead, so there they are graded on their own.

## Seeing it on the robot

Each exercise has a demo that runs your code on a simulated xArm7, the arm used
in the lab. It opens in your browser.

```bash
python local_autograder.py -p vis1    # ex1: your frames and J·θ̇, drawn over the arm
python local_autograder.py -p vis2    # ex2: drag a goal; the arm goes where your IK says
python local_autograder.py -p vis3    # ex3: your joint-space and workspace trajectories, traced
python local_autograder.py -p vis4    # ex4: your gravity torques next to the simulator's
```

Press ctrl-c in the terminal to stop. The demos work before you have written
anything: the arm still moves, and the terminal tells you which function isn't
ready yet.

`python ex4/grav_comp.py` runs ex4's own check: the arm is moved through three
wrist angles, and your torques are compared with what it measures. It always
runs in the simulator. The real xArm7 has no joint torque sensors, so there is
nothing on it to compare against.

## Submitting

```bash
python create_submission.py
```

This writes `handin.zip`. Upload it to Gradescope.
