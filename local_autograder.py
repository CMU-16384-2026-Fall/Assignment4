"""Autograder for HW4: Robot FK and Jacobians (ex1), inverse kinematics (ex2),
trajectories (ex3) and gravity compensation (ex4).

    python local_autograder.py             # grade everything
    python local_autograder.py -p ex2      # one exercise
    python local_autograder.py -p vis1     # run ex1 on the simulated xArm7

The graded modes need only numpy. The vis modes need the course's robot
library, installed from the copy in this handout:
    cd 16384-robot-lib && conda env update -f environment.yml && cd ..
    conda activate 16384
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
TOL = 1e-4
# How close to the reference's joint angles an IK answer (or a workspace
# trajectory built from one) has to land to count as the same answer. Looser
# than TOL because a finite-difference Jacobian steers gradient descent down a
# very slightly different path; anything that lands elsewhere is still judged
# by how close it gets the end effector to the goal.
IK_THETA_TOL = 1e-2

MAX_FAILURES_SHOWN = 10
MAX_DETAIL_CHARS = 160


def _load_module(path):
    """Import a standalone script by path, with its own folder first on sys.path.

    ex1, ex2 and ex3 each have their own robot.py, and the scripts next to
    them do `from robot import Robot`. Which file that finds depends on
    sys.path *and* on whatever is already cached as "robot" in sys.modules, so
    the cache is cleared on the way in and out: each folder gets its own copy.
    """
    folder = str(Path(path).resolve().parent)
    sys.modules.pop("robot", None)
    sys.path.insert(0, folder)
    try:
        spec = importlib.util.spec_from_file_location(Path(path).stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(folder)
        sys.modules.pop("robot", None)
    return module


def _read_jsonl(path):
    """Stream one JSON object per line."""
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _close(actual, expected, tol=TOL):
    """np.allclose that fails (rather than raising) on garbage/missing values.

    Shapes must match exactly: allclose alone would broadcast a scalar or a
    (3, 1) against a (3,) and call it a match.
    """
    try:
        actual = np.asarray(actual, dtype=float)
        expected = np.asarray(expected, dtype=float)
        return actual.shape == expected.shape and bool(
            np.allclose(actual, expected, atol=tol, rtol=0)
        )
    except (TypeError, ValueError):
        return False


def _array(value, name, shape):
    """`value` as a float array of `shape`, or a ValueError saying what it was."""
    if value is None:
        raise ValueError(f"{name} is None")
    array = np.asarray(value, dtype=float)
    if array.shape != tuple(shape):
        raise ValueError(f"{name} has shape {array.shape}, expected {tuple(shape)}")
    return array


def _describe(err):
    return f"{type(err).__name__}: {err}"


def jacobian_tolerance(link_lengths, scale=1.0):
    """Absolute tolerance for anything computed from a numerical Jacobian.

    A forward difference with step h is off by at most (h / 2) * |d2p/dtheta2|,
    and the second derivative of a planar chain's position is bounded by its
    total reach. With the stub's h = 0.001 that is 5e-4 * reach; this allows
    twice that. `scale` multiplies it for things built on J, like J @ theta_dot
    or J^T @ wrench.
    """
    reach = max(1.0, float(np.sum(np.abs(link_lengths))))
    return 1e-3 * reach * max(1.0, float(scale))


def _position_tolerance(link_lengths):
    return TOL * max(1.0, float(np.sum(np.abs(link_lengths))))


def make_robot(robot_cls, link_lengths):
    """A Robot with these link lengths and unit masses (the masses are unused)."""
    n = len(link_lengths)
    return robot_cls(np.array(link_lengths, dtype=float), np.ones(n), np.ones(n), 1.0)


def ee_error(robot, thetas, goal):
    """How far `thetas` leaves `robot`'s end effector from `goal`."""
    goal = np.asarray(goal, dtype=float)
    ee = np.asarray(robot.end_effector(np.asarray(thetas, dtype=float)), dtype=float)
    return float(np.linalg.norm(ee[:goal.shape[0]] - goal))


class Grader:
    """Counts cases and keeps only the failures it is going to print."""

    def __init__(self, name, keep=MAX_FAILURES_SHOWN):
        self.name = name
        self.passed = 0
        self.total = 0
        self.keep = keep
        self.failures = []  # (label, detail), capped at `keep`

    def record(self, label, passed, detail=""):
        self.total += 1
        if passed:
            self.passed += 1
        elif len(self.failures) < self.keep:
            self.failures.append((label, detail))

    def run(self, cases, label, check):
        """Record `check(case, expected)` -> (ok, detail) for every case.

        Anything the student's code raises is a failure of that case, not of
        the run.
        """
        for i, (case, expected) in enumerate(cases):
            try:
                ok, detail = check(case, expected)
            except Exception as err:
                ok, detail = False, _describe(err)
            self.record(label(i, case), ok, "" if ok else detail)
        return self

    def report(self, max_failures=MAX_FAILURES_SHOWN):
        """Print the result: a passing run is one line, a failing one lists the
        failures it kept and says how many it didn't."""
        print(f"\n=== {self.name}: {self.passed}/{self.total} ===")

        shown = min(max_failures, len(self.failures))
        for label, detail in self.failures[:shown]:
            line = f"  [FAIL] {label}"
            if detail:
                if len(detail) > MAX_DETAIL_CHARS:
                    detail = detail[:MAX_DETAIL_CHARS] + "..."
                line += f" — {detail}"
            print(line)

        hidden = (self.total - self.passed) - shown
        if hidden > 0:
            print(f"  ... and {hidden} more failure(s); pass --verbose to keep more")
        return self.passed, self.total


def _chain_label(i, case):
    return f"case {i} ({len(case['link_lengths'])}R)"


# ----------------------------------------------------------------------
# The tests. Each takes the student's code and an iterable of
# (case, expected) pairs, so Gradescope's grade.py runs exactly these on its
# perturbed cases. Where a test needs a Robot to call (the ex1 scripts, ex3's
# workspace trajectory) or to judge an answer by (IK), it is a parameter:
# here it is your own robot.py, on Gradescope the reference one.
# ----------------------------------------------------------------------


def grade_fk(robot_cls, cases, keep=MAX_FAILURES_SHOWN):
    def check(case, expected):
        n = len(case["link_lengths"])
        robot = make_robot(robot_cls, case["link_lengths"])
        frames = _array(robot.forward_kinematics(np.array(case["thetas"], dtype=float)),
                        "frames", (3, 3, n + 1))
        want = np.asarray(expected["frames"])
        tol = _position_tolerance(case["link_lengths"])
        for i in range(n + 1):
            if not _close(frames[:, :, i], want[:, :, i], tol):
                which = "the end effector" if i == n else f"the base of link {i}"
                return False, f"frames[:, :, {i}] ({which}) is wrong"
        return True, ""

    return Grader("ex1: Robot.forward_kinematics", keep).run(cases, _chain_label, check)


def grade_jacobians(robot_cls, cases, keep=MAX_FAILURES_SHOWN):
    def check(case, expected):
        n = len(case["link_lengths"])
        robot = make_robot(robot_cls, case["link_lengths"])
        J = _array(robot.jacobians(np.array(case["thetas"], dtype=float)),
                   "jacobians", (3, n, n + 1))
        want = np.asarray(expected["jacobians"])
        tol = jacobian_tolerance(case["link_lengths"])
        for i in range(n + 1):
            for row, name in enumerate(("x", "y", "theta")):
                if not _close(J[row, :, i], want[row, :, i], tol):
                    return False, (f"jacobians[{row}, :, {i}] (the {name} row for frame {i}) "
                                   f"is off by {np.max(np.abs(J[row, :, i] - want[row, :, i])):.4g}")
        return True, ""

    return Grader("ex1: Robot.jacobians", keep).run(cases, _chain_label, check)


def grade_kinematics(fn, robot_cls, cases, keep=MAX_FAILURES_SHOWN):
    def check(case, expected):
        theta = np.array(case["theta"], dtype=float)
        theta_dot = np.array(case["theta_dot"], dtype=float)
        n = theta.shape[0]
        result = fn(make_robot(robot_cls, case["link_lengths"]), theta.copy(), theta_dot.copy())
        if not isinstance(result, (tuple, list)) or len(result) != 4:
            return False, "expected compute_kinematics to return (x, y, vx, vy)"
        x, y, vx, vy = (_array(v, name, (n,)) for v, name in zip(result, ("x", "y", "vx", "vy")))
        position_tol = _position_tolerance(case["link_lengths"])
        velocity_tol = jacobian_tolerance(
            case["link_lengths"], np.max(np.sum(np.abs(theta_dot), axis=1))
        )
        for name, got, tol in (("x", x, position_tol), ("y", y, position_tol),
                               ("vx", vx, velocity_tol), ("vy", vy, velocity_tol)):
            want = np.asarray(expected[name])
            if not _close(got, want, tol):
                step = int(np.argmax(np.abs(got - want)))
                return False, f"{name} is wrong, worst at time step {step}"
        return True, ""

    def label(i, case):
        return f"case {i} ({len(case['link_lengths'])}R, {len(case['theta'])} steps)"

    return Grader("ex1: sample_velocities.compute_kinematics", keep).run(cases, label, check)


def grade_torques(fn, robot_cls, cases, keep=MAX_FAILURES_SHOWN):
    def check(case, expected):
        n = len(case["link_lengths"])
        wrench = np.array(case["wrench"], dtype=float)
        tau = _array(
            fn(make_robot(robot_cls, case["link_lengths"]),
               np.array(case["thetas"], dtype=float), wrench.copy()),
            "Tau", (n,),
        )
        tol = jacobian_tolerance(case["link_lengths"], np.sum(np.abs(wrench)))
        if not _close(tau, expected["tau"], tol):
            return False, f"Tau is off by {np.max(np.abs(tau - np.asarray(expected['tau']))):.4g}"
        return True, ""

    return Grader("ex1: jacobian_example.compute_joint_torques", keep).run(
        cases, _chain_label, check
    )


def grade_ik(robot_cls, judge_cls, cases, keep=MAX_FAILURES_SHOWN):
    """IK passes if it lands where the reference does, or gets at least as close.

    The second clause is what lets a correct solution that orders the loop a
    little differently (say, checks the stopping condition before updating)
    pass: it ends somewhere else, but no further from the goal. `judge_cls`
    measures that distance.
    """
    def check(case, expected):
        n = len(case["link_lengths"])
        goal = np.array(case["goal"], dtype=float)
        robot = make_robot(robot_cls, case["link_lengths"])
        thetas = _array(
            robot.inverse_kinematics(np.array(case["initial_thetas"], dtype=float), goal.copy()),
            "the returned joint angles", (n,),
        )
        if _close(thetas, expected["thetas"], IK_THETA_TOL):
            return True, ""
        error = ee_error(make_robot(judge_cls, case["link_lengths"]), thetas, goal)
        if error <= expected["ee_error"] + jacobian_tolerance(case["link_lengths"]):
            return True, ""
        return False, (f"ends {error:.4f} from the goal; 100 iterations of the "
                       f"specified descent get within {expected['ee_error']:.4f}")

    def label(i, case):
        kind = "[x, y, theta]" if len(case["goal"]) == 3 else "[x, y]"
        return f"case {i} ({len(case['link_lengths'])}R, {kind} goal)"

    return Grader("ex2: Robot.inverse_kinematics", keep).run(cases, label, check)


def grade_joint_trajectory(fn, cases, keep=MAX_FAILURES_SHOWN):
    def check(case, expected):
        shape = (len(case["start_theta"]), case["num_points"])
        trajectory = _array(
            fn(np.array(case["start_theta"], dtype=float),
               np.array(case["goal_theta"], dtype=float), case["num_points"]),
            "trajectory", shape,
        )
        if not _close(trajectory, expected["trajectory"]):
            return False, "trajectory doesn't interpolate linearly from start to goal"
        return True, ""

    def label(i, case):
        return f"case {i} ({len(case['start_theta'])} joints, {case['num_points']} points)"

    return Grader("ex3: linear_joint_trajectory", keep).run(cases, label, check)


def grade_workspace_trajectory(fn, robot_cls, judge_cls, cases, keep=MAX_FAILURES_SHOWN):
    """Passes on matching the reference, or on keeping every point on the line.

    As with IK, the second clause is judged against what the reference itself
    achieves at that point, since 100 iterations don't always converge.
    """
    def check(case, expected):
        n, k = len(case["link_lengths"]), case["num_points"]
        start = np.array(case["start_theta"], dtype=float)
        goal = np.array(case["goal_pos"], dtype=float)
        trajectory = _array(
            fn(make_robot(robot_cls, case["link_lengths"]), start.copy(), goal.copy(), k),
            "trajectory", (n, k),
        )
        if not _close(trajectory[:, 0], start):
            return False, "the first column should be start_theta"
        if _close(trajectory, expected["trajectory"], IK_THETA_TOL):
            return True, ""

        judge = make_robot(judge_cls, case["link_lengths"])
        begin = np.asarray(judge.end_effector(start), dtype=float)[:len(goal)]
        path = np.array([np.linspace(begin[i], goal[i], k) for i in range(len(goal))])
        tol = jacobian_tolerance(case["link_lengths"])
        for col in range(k):
            error = ee_error(judge, trajectory[:, col], path[:, col])
            if error > expected["ee_errors"][col] + tol:
                return False, (f"column {col} puts the end effector {error:.4f} off the "
                               f"straight line (IK gets within {expected['ee_errors'][col]:.4f})")
        return True, ""

    def label(i, case):
        return f"case {i} ({len(case['link_lengths'])}R, {case['num_points']} points)"

    return Grader("ex3: linear_workspace_trajectory", keep).run(cases, label, check)


def _grav_label(i, case):
    return f"case {i} (theta_a={case['theta_a']:.3f}, theta_b={case['theta_b']:.3f})"


def grade_grav_jacobians(module, cases, keep=MAX_FAILURES_SHOWN):
    def check(case, expected):
        result = module.compute_jacobians(case["theta_a"], case["theta_b"])
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            return False, "expected compute_jacobians to return (J_A, J_B)"
        for name, got in zip(("J_A", "J_B"), result):
            if not _close(_array(got, name, (2, 2)), expected[name]):
                return False, f"{name} is wrong"
        return True, ""

    return Grader("ex4: grav_comp.compute_jacobians", keep).run(cases, _grav_label, check)


def grade_grav_torques(module, cases, keep=MAX_FAILURES_SHOWN):
    def check(case, expected):
        torques = _array(module.compute_gravitational_torques(case["theta_a"], case["theta_b"]),
                         "torques", (2,))
        if not _close(torques, expected["torques"]):
            return False, f"got {np.round(torques, 4).tolist()}"
        return True, ""

    return Grader("ex4: grav_comp.compute_gravitational_torques", keep).run(
        cases, _grav_label, check
    )


# ----------------------------------------------------------------------
# Local runs: your code, the published fixtures
# ----------------------------------------------------------------------


def _fixture(exercise, k):
    """(case, expected) pairs from exN/inK.txt and exN/outK.txt."""
    return zip(_read_jsonl(ROOT / exercise / f"in{k}.txt"),
               _read_jsonl(ROOT / exercise / f"out{k}.txt"))


def _missing(exercise, k):
    paths = [ROOT / exercise / f"in{k}.txt", ROOT / exercise / f"out{k}.txt"]
    return [str(p) for p in paths if not p.exists()]


def _load(name, relpath, *attrs):
    """(values, None) for the named attributes of a student file, or
    (None, a Grader that failed saying why)."""
    try:
        module = _load_module(ROOT / relpath)
    except Exception as err:
        failed = Grader(name)
        failed.record(f"{relpath} imports", False, _describe(err))
        return None, failed
    if not attrs:
        return (module,), None
    missing = [a for a in attrs if not hasattr(module, a)]
    if missing:
        failed = Grader(name)
        failed.record(f"{', '.join(missing)} defined", False, f"not found in {relpath}")
        return None, failed
    return tuple(getattr(module, a) for a in attrs), None


def _run(name, relpath, attrs, exercise, k, grade, keep):
    """Load the student's code, then grade it on exercise/in{k}.txt."""
    missing = _missing(exercise, k)
    if missing:
        failed = Grader(name, keep)
        failed.record("fixtures present", False, "missing " + ", ".join(missing))
        return failed
    loaded, failed = _load(name, relpath, *attrs)
    if failed is not None:
        return failed
    return grade(*loaded, _fixture(exercise, k), keep)


def graders_ex1(keep):
    robot = _load("ex1: Robot", "ex1/robot.py", "Robot")[0]
    robot_cls = robot[0] if robot else None

    def with_robot(name, grade):
        # The ex1 scripts are graded by calling them with your own ex1 Robot,
        # so they can only pass once it does. (Gradescope hands them the
        # reference Robot instead, so there they are graded on their own.)
        def run(fn, cases, keep):
            if robot_cls is None:
                g = Grader(name, keep)
                g.record("ex1/robot.py imports", False, "can't test without a Robot")
                return g
            return grade(fn, robot_cls, cases, keep)
        return run

    return [
        _run("ex1: Robot.forward_kinematics", "ex1/robot.py", ["Robot"], "ex1", 1,
             grade_fk, keep),
        _run("ex1: Robot.jacobians", "ex1/robot.py", ["Robot"], "ex1", 1,
             grade_jacobians, keep),
        _run("ex1: sample_velocities.compute_kinematics", "ex1/sample_velocities.py",
             ["compute_kinematics"], "ex1", 2,
             with_robot("ex1: sample_velocities.compute_kinematics", grade_kinematics), keep),
        _run("ex1: jacobian_example.compute_joint_torques", "ex1/jacobian_example.py",
             ["compute_joint_torques"], "ex1", 3,
             with_robot("ex1: jacobian_example.compute_joint_torques", grade_torques), keep),
    ]


def graders_ex2(keep):
    # IK answers that don't match the reference are judged with your own
    # forward kinematics here, and with the reference's on Gradescope.
    return [
        _run("ex2: Robot.inverse_kinematics", "ex2/robot.py", ["Robot"], "ex2", 1,
             lambda cls, cases, keep: grade_ik(cls, cls, cases, keep), keep),
    ]


def graders_ex3(keep):
    robot = _load("ex3: Robot", "ex3/robot.py", "Robot")[0]
    robot_cls = robot[0] if robot else None

    def workspace(fn, cases, keep):
        if robot_cls is None:
            g = Grader("ex3: linear_workspace_trajectory", keep)
            g.record("ex3/robot.py imports", False, "can't test without a Robot")
            return g
        return grade_workspace_trajectory(fn, robot_cls, robot_cls, cases, keep)

    return [
        _run("ex3: linear_joint_trajectory", "ex3/linear_joint_trajectory.py",
             ["linear_joint_trajectory"], "ex3", 1, grade_joint_trajectory, keep),
        _run("ex3: linear_workspace_trajectory", "ex3/linear_workspace_trajectory.py",
             ["linear_workspace_trajectory"], "ex3", 2, workspace, keep),
    ]


def graders_ex4(keep):
    return [
        _run("ex4: grav_comp.compute_jacobians", "ex4/grav_comp.py", [], "ex4", 1,
             grade_grav_jacobians, keep),
        _run("ex4: grav_comp.compute_gravitational_torques", "ex4/grav_comp.py", [], "ex4", 1,
             grade_grav_torques, keep),
    ]


# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------


def _meshcat_vis():
    """Import the visualization module, or explain what's missing.

    The graded modes need only numpy, so a student can be grading happily and
    still be one `conda env create` short of the simulator.
    """
    try:
        import meshcat_vis
    except ImportError as err:
        sys.exit(
            f"Visualization needs the course's robot library, and it isn't "
            f"importable: {err}\n"
            "Install it from the copy in this handout:\n"
            "    cd 16384-robot-lib && conda env update -f environment.yml && cd ..\n"
            "    conda activate 16384"
        )
    return meshcat_vis


def visualize_ex1():
    """Drive the arm from sliders, drawing your FK frames and J @ theta_dot."""
    robot = _load_module(ROOT / "ex1" / "robot.py")
    _meshcat_vis().visualize_ex1(robot.Robot)


def visualize_ex2():
    """Pick a goal with sliders; the arm goes wherever your IK says."""
    robot = _load_module(ROOT / "ex2" / "robot.py")
    _meshcat_vis().visualize_ex2(robot.Robot)


def visualize_ex3():
    """Run your joint-space and workspace trajectories on the arm."""
    joint = _load_module(ROOT / "ex3" / "linear_joint_trajectory.py")
    workspace = _load_module(ROOT / "ex3" / "linear_workspace_trajectory.py")
    robot = _load_module(ROOT / "ex3" / "robot.py")
    _meshcat_vis().visualize_ex3(
        joint.linear_joint_trajectory, workspace.linear_workspace_trajectory, robot.Robot
    )


def visualize_ex4():
    """Pose the arm's elbow and wrist; compare your torques with the simulator's."""
    _meshcat_vis().visualize_ex4(_load_module(ROOT / "ex4" / "grav_comp.py"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--problem", type=str, default="all",
                        choices=["ex1", "ex2", "ex3", "ex4",
                                 "vis1", "vis2", "vis3", "vis4", "all"])
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="list every failure, not just the first few")
    parser.add_argument("--max-failures", type=int, default=MAX_FAILURES_SHOWN,
                        help="failures to list per test (default: %(default)s)")
    args = parser.parse_args()

    # The visualization modes show, they don't score; they never reach the
    # report below.
    visualizers = {"vis1": visualize_ex1, "vis2": visualize_ex2,
                   "vis3": visualize_ex3, "vis4": visualize_ex4}
    if args.problem in visualizers:
        visualizers[args.problem]()
        return

    keep = 10 ** 9 if args.verbose else args.max_failures
    max_failures = keep if args.verbose else args.max_failures
    exercises = {"ex1": graders_ex1, "ex2": graders_ex2,
                 "ex3": graders_ex3, "ex4": graders_ex4}
    selected = list(exercises) if args.problem == "all" else [args.problem]

    total_passed = total_count = 0
    for name in selected:
        for grader in exercises[name](keep):
            p, t = grader.report(max_failures=max_failures)
            total_passed += p
            total_count += t

    print(f"\n=== TOTAL: {total_passed}/{total_count} ===")
    sys.exit(0 if total_passed == total_count else 1)


if __name__ == "__main__":
    main()
