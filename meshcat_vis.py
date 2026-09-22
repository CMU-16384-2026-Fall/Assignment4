"""Run the students' HW4 code on the simulated xArm7, in meshcat.

ex1-ex3 use the arm as a planar 3R: joints 1, 4 and 7 with their axes all
parallel to world z, the rest held. Those held joints are commanded every
tick, not welded in the model: the arm stays a real 7-DoF arm that the safety
guard and the servos still understand, it just isn't asked to move them.
`PlanarChain` is the map between the exercise's planar angles (`theta`) and the
arm's seven joint values (`q`), and it measures itself off the loaded model
rather than carrying the numbers as constants.

ex4 uses the arm in the vertical plane instead (joints 4 and 6), which is the
model ex4/grav_comp.py describes.

Nothing in here grades anything. `local_autograder.py` decides right and wrong
from the fixtures; this only shows you what your code does.
"""

import collections
import http.server
import json
import math
import socket
import string
import sys
import threading
import time
import urllib.parse
import webbrowser

import meshcat.geometry as g
import mujoco
import numpy as np
from xarm7_lib import FREE_JOINTS, HOME_POSE, SimulatedXArm7
from xarm7_lib.mujoco_meshcat import MeshcatVisualizer

# The lab pose, from the course library rather than restated here: joints 1, 4
# and 7 (0-based 0, 3, 6 -> theta1, theta2, theta3) are free, and the other
# four are held at HOME_POSE's values (joints 2, 3 and 6 at +pi/2, joint 5 at
# -pi/2), which leaves the free joints' axes all parallel to world z.
LOCKED_JOINTS = {i: float(HOME_POSE[i]) for i in range(7) if i not in FREE_JOINTS}

DEFAULT_TOOL_LENGTH = 0.10  # m, the drawn third link

# Everything this module draws lives under here, clear of the "mujoco" and
# "safety" prefixes `MeshcatVisualizer` publishes under.
PREFIX = "rrr"

_ARM_COLOR = 0xFFC53D  # the student's FK, drawn over the arm
_TOOL_COLOR = 0xFF5A5F  # the virtual third link
_CLOUD_COLOR = (0.32, 0.42, 0.58)  # the whole predicted workspace
_TRACE_COLOR = (1.00, 0.77, 0.24)  # the part of it the arm has visited

_JOINT_MARKER_RADIUS = 0.018
_TRIAD_SCALE = 0.12
_CLOUD_POINT_SIZE = 0.006
_TRACE_POINT_SIZE = 0.014

_PROBE_DELTA = 0.3  # rad, the step each free joint is probed with

# ----------------------------------------------------------------------
# The planar arm hiding inside the xArm7
# ----------------------------------------------------------------------


def _wrap(angle):
    """Fold an angle into (-pi, pi]."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


class PlanarChain:
    """The planar RRR the locked xArm7 is, and the map to and from its joints.

    Every field is measured from the loaded model by `probe`, so this holds no
    geometry of its own. For each free joint i the map is

        theta_i = sign_i * q_i + offset_i

    with `sign_i` +-1 (joints 4 and 7 turn about -z once the arm is locked, so
    theirs are negative) and `offset_i` whatever the locked pose builds in.
    """

    def __init__(self, lengths, signs, offsets, limits, heights):
        self.link_lengths = [float(v) for v in lengths]  # [l1, l2, l3]
        self._signs = np.asarray(signs, dtype=float)
        self._offsets = np.asarray(offsets, dtype=float)
        self._limits = np.asarray(limits, dtype=float)  # (3, 2), in theta
        # z of the shoulder/elbow plane and of the wrist/tool plane. The wrist
        # sits ~97 mm below the elbow, which costs the arm nothing — the three
        # axes are still parallel, so the projection onto xy is exactly planar
        # — but the drawing has to know, or the shadow floats off the arm.
        self.plane_z, self.tool_z = (float(v) for v in heights)

    # -- construction --------------------------------------------------

    @classmethod
    def probe(cls, arm, tool_length=DEFAULT_TOOL_LENGTH):
        """Measure the chain off `arm`'s MuJoCo model.

        Runs `mj_forward` on a scratch `MjData` rather than on the arm's own,
        so the simulation thread never sees the probe poses.
        """
        pose = _Prober(arm)
        home = locked_configuration()

        origins, axes, _ = pose.at(home)
        if not np.allclose(np.abs(axes[:, 2]), 1.0, atol=1e-6):
            raise RuntimeError(
                "the locked pose does not leave a planar arm: joint axes "
                f"{np.round(axes, 4).tolist()} are not all parallel to z"
            )

        shoulder, elbow, wrist = origins
        l1 = float(np.linalg.norm((elbow - shoulder)[:2]))
        l2 = float(np.linalg.norm((wrist - elbow)[:2]))

        # theta_k is segment k's heading relative to segment k-1. The headings
        # at the home pose give the offsets; stepping one joint at a time gives
        # each sign. Both are measured, neither is assumed.
        home_headings = pose.headings(home)
        signs, offsets = [], []
        for k, index in enumerate(FREE_JOINTS):
            stepped = home.copy()
            stepped[index] += _PROBE_DELTA
            moved = pose.headings(stepped)[k]
            # Only segment k and those downstream turn, so the change in the
            # relative angle is the change in this segment's own heading.
            signs.append(math.copysign(1.0, _wrap(moved - home_headings[k])))
            previous = home_headings[k - 1] if k else 0.0
            offsets.append(
                _wrap((home_headings[k] - previous) - signs[k] * home[index])
            )

        limits = []
        for k, index in enumerate(FREE_JOINTS):
            low, high = arm.position_limit[index]
            limits.append(sorted(signs[k] * np.array([low, high]) + offsets[k]))

        return cls(
            lengths=[l1, l2, float(tool_length)],
            signs=signs,
            offsets=offsets,
            limits=limits,
            heights=(shoulder[2], wrist[2]),
        )

    # -- the map -------------------------------------------------------

    def q_from_theta(self, thetas):
        """The seven joint values realizing these three planar angles.

        Clamped to the theta range the arm's joint limits allow, so a slider
        sitting on an endpoint can't round its way outside and have
        `set_joint_targets` refuse the pose.
        """
        thetas = np.clip(
            np.asarray(thetas, dtype=float).reshape(3),
            self._limits[:, 0],
            self._limits[:, 1],
        )
        q = locked_configuration()
        for k, index in enumerate(FREE_JOINTS):
            q[index] = (thetas[k] - self._offsets[k]) / self._signs[k]
        return q

    def theta_from_q(self, q):
        """The three planar angles `q` is at."""
        q = np.asarray(q, dtype=float).reshape(7)
        return np.array(
            [self._signs[k] * q[i] + self._offsets[k] for k, i in enumerate(FREE_JOINTS)]
        )

    def theta_limits(self):
        """(minJoint, maxJoint) in theta, from the arm's own joint ranges."""
        return self._limits[:, 0].copy(), self._limits[:, 1].copy()

    def vertices(self, thetas):
        """Shoulder, elbow, wrist and tool tip in world coordinates.

        Each at the height it actually sits at, so a correct FK draws itself
        onto the arm instead of hovering over it.
        """
        t1, t2, t3 = (float(v) for v in thetas)
        l1, l2, l3 = self.link_lengths
        a1, a2, a3 = t1, t1 + t2, t1 + t2 + t3
        shoulder = np.array([0.0, 0.0, self.plane_z])
        elbow = shoulder + [l1 * math.cos(a1), l1 * math.sin(a1), 0.0]
        wrist = elbow + [l2 * math.cos(a2), l2 * math.sin(a2), self.tool_z - self.plane_z]
        tip = wrist + [l3 * math.cos(a3), l3 * math.sin(a3), 0.0]
        return np.array([shoulder, elbow, wrist, tip])

    def describe(self):
        l1, l2, l3 = self.link_lengths
        return (
            f"planar RRR: l1={l1:.4f} m  l2={l2:.4f} m  l3={l3:.4f} m (drawn tool)"
            f"\n  arm plane z={self.plane_z:.3f} m, wrist plane z={self.tool_z:.3f} m"
        )


class _Prober:
    """Forward kinematics on a scratch state, for measuring the chain.

    Kept off `arm.data` on purpose: the simulation thread is reading that
    thirty times a second and the probe poses are not poses the arm is in.
    """

    def __init__(self, arm):
        self.arm = arm
        self.model = arm.model
        self.data = mujoco.MjData(arm.model)

    def at(self, q):
        """(origins, axes, tool heading) for the three free joints at `q`."""
        self.data.qpos[:] = 0.0
        self.data.qpos[self.arm._qadr] = q
        mujoco.mj_forward(self.model, self.data)

        origins, axes = [], []
        for index in FREE_JOINTS:
            jid = self.arm._jid[index]
            R = self.data.xmat[self.model.jnt_bodyid[jid]].reshape(3, 3)
            origins.append(
                self.data.xpos[self.model.jnt_bodyid[jid]]
                + R @ self.model.jnt_pos[jid]
            )
            axes.append(R @ self.model.jnt_axis[jid])

        # The drawn tool points along link7's own x axis, so that is the third
        # segment's heading.
        wrist_R = self.data.xmat[
            self.model.jnt_bodyid[self.arm._jid[FREE_JOINTS[2]]]
        ].reshape(3, 3)
        tool = math.atan2(wrist_R[1, 0], wrist_R[0, 0])
        return np.array(origins), np.array(axes), tool

    def headings(self, q):
        """World heading of each of the three planar segments at `q`, in rad."""
        origins, _, tool = self.at(q)
        return [
            math.atan2(*(origins[1] - origins[0])[[1, 0]]),
            math.atan2(*(origins[2] - origins[1])[[1, 0]]),
            tool,
        ]


def locked_configuration(thetas_free=None):
    """A seven-vector with the four held joints filled in.

    `thetas_free` is optional raw joint values for joints 1, 4 and 7.
    """
    q = np.zeros(7)
    for index, value in LOCKED_JOINTS.items():
        q[index] = value
    if thetas_free is not None:
        for k, index in enumerate(FREE_JOINTS):
            q[index] = float(thetas_free[k])
    return q


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------


def _node(viz, path):
    return viz.viewer[f"{PREFIX}/{path}"]


def draw_polyline(viz, path, points, color=_ARM_COLOR, width=3.0):
    """Draw an open polyline through `points`, an (n, 3) array."""
    points = np.asarray(points, dtype=np.float32).T  # meshcat wants (3, n)
    with viz.lock:
        _node(viz, path).set_object(
            g.Line(
                g.PointsGeometry(points),
                g.LineBasicMaterial(color=color, linewidth=width),
            )
        )


# Marker sets that have already been published, so a redraw only has to move
# them. These are drawn every frame, and re-sending the spheres each time would
# be a dozen ZMQ round trips per frame, taken against the same viewer lock the
# simulation thread needs to sync.
_MARKERS = {}


def draw_markers(viz, path, points, radius=_JOINT_MARKER_RADIUS, color=_ARM_COLOR):
    """Draw a small sphere at each of `points`, an (n, 3) array."""
    points = np.asarray(points, dtype=float)
    key = (id(viz), path)
    shape = (len(points), float(radius), int(color))

    with viz.lock:
        if _MARKERS.get(key) != shape:
            _node(viz, path).delete()
            material = g.MeshLambertMaterial(color=color)
            for i in range(len(points)):
                _node(viz, f"{path}/{i}").set_object(g.Sphere(radius), material)
            _MARKERS[key] = shape
        for i, point in enumerate(points):
            T = np.eye(4)
            T[:3, 3] = point
            _node(viz, f"{path}/{i}").set_transform(T)


def draw_points(viz, path, xy, z, color=_CLOUD_COLOR, size=_CLOUD_POINT_SIZE):
    """Draw an (n, 2) set of planar points as a cloud at height `z`."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if xy.size == 0:
        return
    position = np.column_stack([xy, np.full(len(xy), float(z))]).T.astype(np.float32)
    colors = np.tile(np.asarray(color, dtype=np.float32).reshape(3, 1), (1, len(xy)))
    with viz.lock:
        _node(viz, path).set_object(
            g.PointCloud(position, colors, size=size)
        )


def draw_triad(viz, path, T, scale=_TRIAD_SCALE):
    """Draw an RGB axis triad at the 4x4 pose `T`."""
    with viz.lock:
        node = _node(viz, path)
        node.set_object(g.triad(scale))
        node.set_transform(np.asarray(T, dtype=float))


def clear(viz, path=""):
    """Remove everything this module drew under `path`."""
    with viz.lock:
        node = _node(viz, path) if path else viz.viewer[PREFIX]
        node.delete()
    for key in [k for k in _MARKERS if k[0] == id(viz) and k[1].startswith(path)]:
        del _MARKERS[key]


# ----------------------------------------------------------------------
# Joint limits the arm can actually sweep
# ----------------------------------------------------------------------


def _one_turn(low, high):
    """Trim a joint range spanning more than 2*pi down to one revolution."""
    if high - low <= 2.0 * math.pi:
        return low, high
    middle = 0.5 * (low + high)
    return middle - math.pi, middle + math.pi


def collision_free_limits(arm, chain, samples=200, wrist_samples=12, margin=0.05):
    """(minJoint, maxJoint) in theta, with self-colliding elbow angles removed.

    theta1 and theta3 are left at the arm's full range: turning the base cannot
    create a self-collision, and the wrist is deliberately unconstrained. Only
    theta2 needs narrowing, and it is narrowed by asking the arm's own guard
    (`check_safety`, the same check that stops the simulated arm) rather than
    by picking a number: scan theta2 across its range, keep the longest run
    that is clear at every wrist angle, then inset by `margin`.

    Checking every wrist angle matters. The forearm folded back over the upper
    link is a collision whatever the wrist does, but nearer the edge of the
    band it is the wrist that touches first, and a band validated at one wrist
    angle would hand the sweep poses the guard then refuses.
    """
    low, high = chain.theta_limits()
    # joint1 and joint7 both turn through more than a full revolution, and a
    # sweep past one adds no workspace — it re-traces what it already drew, at
    # twice the cost. theta2 has well under a turn to give and is left alone.
    low[0], high[0] = _one_turn(low[0], high[0])
    low[2], high[2] = _one_turn(low[2], high[2])

    grid = np.linspace(low[1], high[1], int(samples))
    wrists = np.linspace(0.0, 2.0 * math.pi, int(wrist_samples), endpoint=False)

    safe = np.array(
        [
            all(arm.check_safety(chain.q_from_theta([0.0, t2, t3])) is None for t3 in wrists)
            for t2 in grid
        ]
    )

    start, best = None, None
    for i, ok in enumerate(np.append(safe, False)):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if best is None or i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    if best is None:
        raise RuntimeError(
            "no self-collision-free elbow angle found; the locked pose may be wrong"
        )

    lo, hi = grid[best[0]], grid[best[1] - 1]
    lo, hi = lo + margin, hi - margin
    if lo >= hi:
        raise RuntimeError(f"the collision-free elbow band is narrower than {margin} rad")

    print(
        f"[vis] elbow band, self-collision free: theta2 in "
        f"[{math.degrees(lo):.1f}, {math.degrees(hi):.1f}] deg "
        f"(of [{math.degrees(low[1]):.1f}, {math.degrees(high[1]):.1f}] reachable)"
    )
    return (
        [float(low[0]), float(lo), float(low[2])],
        [float(high[0]), float(hi), float(high[2])],
    )


# ----------------------------------------------------------------------
# The slider page
# ----------------------------------------------------------------------

# `string.Template` rather than `str.format`: the page is mostly CSS and JS, and
# `.format` would need every brace in it doubled.
_PAGE = string.Template("""<!doctype html>
<title>$title</title>
<style>
  body { margin: 0; font: 14px system-ui, sans-serif; background: #14171c; color: #e8ecf1; }
  #panel { padding: 10px 16px 12px; border-bottom: 1px solid #2b313a; }
  .row { display: flex; align-items: center; gap: 12px; margin: 6px 0; }
  .row label { width: 5.5em; color: #9aa7b8; }
  .row input { flex: 1; }
  .row output { width: 7em; text-align: right; font-variant-numeric: tabular-nums; }
  #status { margin-top: 8px; color: #9aa7b8; font-variant-numeric: tabular-nums; }
  iframe { display: block; width: 100vw; height: calc(100vh - $chrome); border: 0; }
  button { background: #2b313a; color: #e8ecf1; border: 0; padding: 4px 12px;
           border-radius: 4px; cursor: pointer; }
</style>
<div id="panel">
$rows
  <div class="row">
    <div id="status">&mdash;</div>
    <button id="reset">reset</button>
  </div>
</div>
<iframe src="$meshcat_url"></iframe>
<script>
  const SPECS = $specs;
  const el = k => document.getElementById(k);

  function show(spec) {
    const v = (+el("s_" + spec.key).value).toFixed(spec.digits);
    el("o_" + spec.key).textContent = v + spec.unit;
  }
  function push() {
    SPECS.forEach(show);
    fetch("set?" + SPECS.map(s => s.key + "=" + el("s_" + s.key).value).join("&"));
  }
  SPECS.forEach(s => el("s_" + s.key).addEventListener("input", push));
  el("reset").addEventListener("click", () => {
    SPECS.forEach(s => el("s_" + s.key).value = s.value);
    push();
  });

  setInterval(async () => {
    const state = await (await fetch("state")).json();
    el("status").innerHTML = state.status;
  }, 250);
  push();
</script>
""")

_ROW = string.Template(
    '  <div class="row">\n'
    '    <label for="s_$key">$label</label>\n'
    '    <input id="s_$key" type="range" min="$low" max="$high" step="$step" value="$value">\n'
    '    <output id="o_$key"></output>\n'
    "  </div>"
)

# Page chrome above the viewer: one row per slider, plus the status row.
_ROW_HEIGHT = 30  # px
_PANEL_PADDING = 40  # px


class SliderSpec(collections.namedtuple(
    "SliderSpec", "key label low high step value unit digits"
)):
    """One slider on the page.

    `unit` and `digits` are only how the browser renders the number back to the
    reader; python always works in the raw value.
    """

    __slots__ = ()

    def __new__(cls, key, label, low, high, step, value, unit="", digits=2):
        return super().__new__(
            cls, key, label, float(low), float(high), float(step),
            float(np.clip(value, low, high)), unit, int(digits),
        )


class SliderPage:
    """Sliders on a local page, with the meshcat viewer embedded below.

    meshcat's own dat.GUI can't be driven from python — its ZMQ bridge forwards
    only the five scene commands and drops anything else, and the browser->python
    direction is wired up for image captures alone — so the controls are served
    from here instead, and the viewer is put in an iframe so it is still one tab.

    Values live behind a lock: `values` is what the caller reads each tick, and
    `set_status` is the line the page echoes back. The page knows nothing about
    what the numbers mean; formatting the status line is the caller's job.
    """

    def __init__(self, specs, meshcat_url, title="xArm7", open_browser=True):
        self._specs = list(specs)
        self._lock = threading.Lock()
        self._values = {s.key: s.value for s in self._specs}
        self._status = "&mdash;"

        page = _PAGE.substitute(
            title=title,
            meshcat_url=meshcat_url,
            chrome=f"{_PANEL_PADDING + _ROW_HEIGHT * (len(self._specs) + 1)}px",
            rows="\n".join(
                _ROW.substitute(
                    key=s.key, label=s.label,
                    low=f"{s.low:.4f}", high=f"{s.high:.4f}",
                    step=f"{s.step:g}", value=f"{s.value:.4f}",
                )
                for s in self._specs
            ),
            specs=json.dumps([
                {"key": s.key, "value": s.value, "unit": s.unit, "digits": s.digits}
                for s in self._specs
            ]),
        ).encode("utf-8")

        self._server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", _free_port()), _handler_for(self, page)
        )
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="rrr-sliders", daemon=True
        )
        self._thread.start()

        host, port = self._server.server_address
        self.url = f"http://{host}:{port}/"
        print(f"[vis] controls: {self.url}")
        if open_browser:
            webbrowser.open(self.url)

    @property
    def values(self):
        """The current slider values, by key."""
        with self._lock:
            return dict(self._values)

    def set_status(self, text):
        """Publish the line the page shows under the sliders."""
        with self._lock:
            self._status = str(text)

    def _accept(self, query):
        parsed = urllib.parse.parse_qs(query)
        with self._lock:
            for spec in self._specs:
                value = parsed.get(spec.key)
                if value:
                    self._values[spec.key] = float(np.clip(
                        float(value[0]), spec.low, spec.high
                    ))

    def _state(self):
        with self._lock:
            return {"values": dict(self._values), "status": self._status}

    def close(self):
        self._server.shutdown()
        self._server.server_close()


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _handler_for(sliders, page):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            route, _, query = self.path.lstrip("/").partition("?")
            if route in ("", "index.html"):
                self._reply("text/html; charset=utf-8", page)
            elif route == "set":
                sliders._accept(query)
                self._reply("text/plain", b"ok")
            elif route == "state":
                body = json.dumps(sliders._state()).encode("utf-8")
                self._reply("application/json", body)
            else:
                self.send_error(404)

        def _reply(self, content_type, body):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass  # a slider drag is a request per frame; don't narrate it

    return Handler


# ----------------------------------------------------------------------
# Shared setup
# ----------------------------------------------------------------------


def start_arm(tool_length=DEFAULT_TOOL_LENGTH, open_browser=True, **kwargs):
    """A visualized xArm7 holding the locked pose, and its `PlanarChain`.

    The safety box is off: it is drawn around the volume the *lab* arm may use,
    and the planar sweep turns the whole way round, so leaving it on would clip
    most of theta1 away. Self-collision checking stays on — that is the
    constraint the exercises care about.

    The visualizer is attached here rather than through `visualize=True` only
    so it can be told not to open a tab of its own: every demo serves the
    viewer inside its own slider page, and two tabs of the same scene is one
    too many.
    """
    options = dict(visualize=False, safety_box=None, guard=True)
    options.update(kwargs)
    arm = SimulatedXArm7(**options)
    try:
        arm.viz = MeshcatVisualizer(arm.model, arm.data, open_browser=open_browser)
        print(f"[vis] meshcat: {arm.viz.url}")
        chain = PlanarChain.probe(arm, tool_length=tool_length)
        print("[vis] " + chain.describe())
        arm.set_joint_targets(locked_configuration(), speed=0.8)
    except Exception:
        arm.close()
        raise
    return arm, chain


def tool_segment(arm, chain):
    """The virtual third link where the arm is right now: (wrist, tip).

    Read from `data` under the arm's lock, since the simulation thread is
    writing it. The tool has no geometry in the model — it is drawn, not
    simulated — so it is placed on link7's own x axis.
    """
    with arm._lock:
        jid = arm._jid[FREE_JOINTS[2]]
        body = arm.model.jnt_bodyid[jid]
        R = arm.data.xmat[body].reshape(3, 3)
        wrist = arm.data.xpos[body] + R @ arm.model.jnt_pos[jid]
    return wrist, wrist + chain.link_lengths[2] * R[:, 0]


def measured_tip(arm, chain):
    """Where the drawn tool tip actually is, from MuJoCo. (x, y)."""
    tip = tool_segment(arm, chain)[1]
    return float(tip[0]), float(tip[1])


def draw_tool(viz, arm, chain):
    """Draw the virtual third link out of the flange, where the arm is now."""
    wrist, tip = tool_segment(arm, chain)
    draw_polyline(viz, "tool", np.array([wrist, tip]), color=_TOOL_COLOR, width=6.0)
    draw_markers(viz, "tool_tip", np.array([tip]), radius=0.012, color=_TOOL_COLOR)


def _xy(point):
    """A tip position for the status line, or a dash if there isn't one."""
    if point is None:
        return "\u2014"
    return f"({point[0]:.3f}, {point[1]:.3f})"


# ----------------------------------------------------------------------
# The student's code, and what to do when it isn't ready yet
# ----------------------------------------------------------------------

_VELOCITY_COLOR = 0x4CC9F0  # the student's J @ theta_dot
_TRUE_VELOCITY_COLOR = 0x7BD88F  # the arm's own tip velocity
_GOAL_COLOR = 0xB388FF
_LINE_COLOR = 0xE8ECF1
_COM_COLOR = 0x4CC9F0
_JOINT_PATH_COLOR = (0.30, 0.79, 0.94)  # traced tip, joint-space trajectory
_WORKSPACE_PATH_COLOR = (1.00, 0.77, 0.24)  # traced tip, workspace trajectory
_PATH_POINT_SIZE = 0.012

# Velocity arrows are drawn as the distance the tip would cover in this long,
# so a 1 m/s tip velocity is a 25 cm arrow: long enough to read, short enough
# to stay on screen at the slider's top speed.
_ARROW_SECONDS = 0.25
_FD_STEP = 1e-6  # rad, for the arm's own tip velocity


class Complaints:
    """Say once, per distinct message, that the student's code isn't usable yet.

    Every demo calls into the student's code every frame, and the stubs raise
    or return zeros until the To-Dos are filled in. The arm keeps moving
    regardless; this only keeps the terminal readable.
    """

    def __init__(self, what, where):
        self.what = what
        self.where = where
        self._last = None

    def __call__(self, err):
        message = err if isinstance(err, str) else f"{type(err).__name__}: {err}"
        if message == self._last:
            return
        self._last = message
        print(f"[vis] {self.what} isn't usable yet — {message}\n"
              f"      The arm still moves; fill in the To-Dos in {self.where} "
              "to see your answer drawn.")

    def clear(self):
        self._last = None


def student_robot(robot_cls, chain):
    """The student's Robot, built with the arm's own planar link lengths."""
    lengths = np.array(chain.link_lengths, dtype=float)
    n = len(lengths)
    return robot_cls(lengths, np.ones(n), np.ones(n), 1.0)


def _checked(value, name, shape):
    """`value` as a float array of `shape`; refuses None, NaN and all-zero stubs."""
    if value is None:
        raise ValueError(f"{name} is None")
    array = np.asarray(value, dtype=float)
    if array.shape != tuple(shape):
        raise ValueError(f"{name} has shape {array.shape}, expected {tuple(shape)}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} has NaN or inf in it")
    if not np.any(array):
        raise ValueError(f"{name} is all zeros (still the stub?)")
    return array


def _lift(chain, points_xy):
    """Planar frame origins placed at the heights the arm's joints sit at.

    Frames 0 and 1 (shoulder, elbow) are in the arm plane; 2 and 3 (wrist,
    tool tip) are ~97 mm lower, in the wrist plane. See `PlanarChain`.
    """
    heights = [chain.plane_z, chain.plane_z, chain.tool_z, chain.tool_z]
    return np.array([[p[0], p[1], z] for p, z in zip(points_xy, heights)])


def draw_arrow(viz, path, start, vector_xy, z, color):
    """A planar arrow from `start` (x, y) along `vector_xy`, at height `z`."""
    start = np.array([start[0], start[1], z], dtype=float)
    end = start + np.array([vector_xy[0], vector_xy[1], 0.0])
    draw_polyline(viz, f"{path}/shaft", np.array([start, end]), color=color, width=5.0)
    draw_markers(viz, f"{path}/head", np.array([end]), radius=0.01, color=color)


def _vxy(v):
    return "—" if v is None else f"({v[0]:+.3f}, {v[1]:+.3f})"


def _serve(rate, tick):
    """Call `tick()` at `rate` Hz until ctrl-c."""
    period = 1.0 / float(rate)
    while True:
        tick()
        time.sleep(period)


# ----------------------------------------------------------------------
# ex1: forward kinematics and Jacobians, on sliders
# ----------------------------------------------------------------------


def visualize_ex1(robot_cls, rate=30.0, open_browser=True):
    """Pose the arm with sliders; draw your frames and your J @ theta_dot over it.

    The arm is driven through the calibrated chain, not through your code, so
    the demo works before any of it does. Your `forward_kinematics` draws the
    skeleton and the end-effector triad; your `jacobians` draws the predicted
    tip velocity (blue) for the theta-dot sliders. The green arrow is the tip
    velocity of the arm itself, from the calibrated chain. When your arrow
    covers the green one, your end-effector Jacobian is right.
    """
    arm, chain = start_arm(open_browser=False)
    page = None
    try:
        low, high = chain.theta_limits()
        page = SliderPage(
            [SliderSpec(f"t{i}", f"theta{i + 1}", low[i], high[i], 0.005, 0.0,
                        " rad", 2) for i in range(3)]
            + [SliderSpec(f"w{i}", f"dtheta{i + 1}", -2.0, 2.0, 0.01,
                          0.5 if i == 0 else 0.0, " rad/s", 2) for i in range(3)],
            arm.viz.url,
            title="Robot FK and Jacobians — xArm7",
            open_browser=open_browser,
        )
        robot = student_robot(robot_cls, chain)
        fk_complaints = Complaints("Robot.forward_kinematics", "ex1/robot.py")
        jac_complaints = Complaints("Robot.jacobians", "ex1/robot.py")
        print("[vis] drag the sliders; ctrl-c here to stop.")

        def tick():
            values = page.values
            q = chain.q_from_theta([values[f"t{i}"] for i in range(3)])
            # The angles the arm is actually asked for, after clamping.
            thetas = chain.theta_from_q(q)
            theta_dot = np.array([values[f"w{i}"] for i in range(3)])
            arm.servo_joints(q)
            draw_tool(arm.viz, arm, chain)

            your_tip = None
            try:
                frames = _checked(robot.forward_kinematics(thetas.copy()), "frames", (3, 3, 4))
                points = _lift(chain, frames[:2, 2, :].T)
                draw_polyline(arm.viz, "fk/links", points)
                draw_markers(arm.viz, "fk/joints", points[:3])
                T = np.eye(4)
                T[:2, :2] = frames[:2, :2, 3]
                T[:3, 3] = points[3]
                draw_triad(arm.viz, "fk/frame", T)
                your_tip = frames[:2, 2, 3]
                fk_complaints.clear()
            except Exception as err:
                fk_complaints(err)
                clear(arm.viz, "fk")

            tip = chain.vertices(thetas)[3][:2]
            true_v = (chain.vertices(thetas + theta_dot * _FD_STEP)[3][:2] - tip) / _FD_STEP
            draw_arrow(arm.viz, "velocity/arm", tip, _ARROW_SECONDS * true_v,
                       chain.tool_z, _TRUE_VELOCITY_COLOR)

            your_v = None
            try:
                J = np.asarray(robot.jacobians(thetas.copy()), dtype=float)
                J = _checked(J, "jacobians", (3, 3, 4))
                your_v = J[:, :, 3] @ theta_dot
                draw_arrow(arm.viz, "velocity/yours",
                           tip if your_tip is None else your_tip,
                           _ARROW_SECONDS * your_v[:2], chain.tool_z, _VELOCITY_COLOR)
                jac_complaints.clear()
            except Exception as err:
                jac_complaints(err)
                clear(arm.viz, "velocity/yours")

            page.set_status(
                f"tip: yours <b>{_xy(your_tip)}</b> arm <b>{_xy(measured_tip(arm, chain))}</b>"
                f" &nbsp; v: yours <b>{_vxy(your_v)}</b> arm <b>{_vxy(true_v)}</b> m/s"
            )

        _serve(rate, tick)
    except KeyboardInterrupt:
        print("\n[vis] stopping.")
    finally:
        if page is not None:
            page.close()
        arm.stop(wait=False)
        arm.close()


# ----------------------------------------------------------------------
# ex2: inverse kinematics, to a goal on sliders
# ----------------------------------------------------------------------


_IK_CALLS = 30  # most restarts of the student's IK per goal
_IK_SETTLED = 1e-4  # rad: an answer this close to the last one has converged
_GOAL_STILL = 0.25  # s the goal must rest before IK runs, so dragging doesn't lag


def visualize_ex2(robot_cls, rate=30.0, speed=1.5, open_browser=True):
    """Set a goal with sliders; the arm moves wherever your IK says to.

    Once the goal has been still for a moment, your `inverse_kinematics` runs
    from the arm's current angles and the arm is sent to the answer. The purple
    marker is the goal (with a heading line when `match angle` is on); the
    status line says how far the arm's real tip ended up from it.

    The descent's step size and 100 iterations were sized for the metre-long
    links of IK_example.py. The xArm's links are a third of that, the gradient
    is correspondingly smaller, and one call gets only part of the way. So
    your IK is called again from its own answer until it stops moving (up to
    `_IK_CALLS` times); the status line says how many calls that took.
    """
    arm, chain = start_arm(open_browser=False)
    page = None
    try:
        reach = float(sum(chain.link_lengths))
        page = SliderPage(
            [
                SliderSpec("x", "goal x", -reach, reach, 0.005, 0.45 * reach, " m", 3),
                SliderSpec("y", "goal y", -reach, reach, 0.005, 0.35 * reach, " m", 3),
                SliderSpec("phi", "goal angle", -math.pi, math.pi, 0.01, 0.0, " rad", 2),
                SliderSpec("match", "match angle", 0, 1, 1, 0, "", 0),
            ],
            arm.viz.url,
            title="Robot inverse kinematics — xArm7",
            open_browser=open_browser,
        )
        robot = student_robot(robot_cls, chain)
        complaints = Complaints("Robot.inverse_kinematics", "ex2/robot.py")
        state = {"goal": None, "changed": 0.0, "solved": True, "note": ""}
        print("[vis] drag the goal; ctrl-c here to stop.")

        def solve(goal):
            start = chain.theta_from_q(arm.joint_values)
            thetas = start
            for calls in range(1, _IK_CALLS + 1):
                answer = np.asarray(robot.inverse_kinematics(thetas.copy(), goal.copy()),
                                    dtype=float)
                if answer.shape != (3,) or not np.all(np.isfinite(answer)):
                    raise ValueError(f"returned {answer!r}, expected 3 finite joint angles")
                if calls == 1 and np.allclose(answer, start):
                    raise ValueError("returned the initial angles unchanged (still the stub?)")
                moved = float(np.max(np.abs(answer - thetas)))
                thetas = answer
                if moved < _IK_SETTLED:
                    break
            q = chain.q_from_theta(thetas)
            note = f" &nbsp; IK called {calls}\u00d7"
            if arm.check_safety(q) is not None:
                note += " &nbsp; <i>that pose self-collides; the arm stops short</i>"
            arm.set_joint_targets(q, speed=speed, wait=False)
            return note

        def tick():
            values = page.values
            match = values["match"] >= 0.5
            goal = np.array([values["x"], values["y"]] + ([values["phi"]] if match else []))
            now = time.monotonic()
            if state["goal"] is None or goal.shape != state["goal"].shape \
                    or not np.allclose(goal, state["goal"]):
                state.update(goal=goal, changed=now, solved=False)
            if not state["solved"] and now - state["changed"] >= _GOAL_STILL:
                state["solved"] = True
                try:
                    state["note"] = solve(goal)
                    complaints.clear()
                except Exception as err:
                    complaints(err)
                    state["note"] = " &nbsp; <i>your IK isn't answering yet</i>"

            z = chain.tool_z
            draw_markers(arm.viz, "goal/point", np.array([[goal[0], goal[1], z]]),
                         radius=0.02, color=_GOAL_COLOR)
            if match:
                heading = 0.08 * np.array([math.cos(goal[2]), math.sin(goal[2])])
                draw_arrow(arm.viz, "goal/heading", goal[:2], heading, z, _GOAL_COLOR)
            else:
                clear(arm.viz, "goal/heading")
            draw_tool(arm.viz, arm, chain)

            tip = np.array(measured_tip(arm, chain))
            page.set_status(
                f"goal <b>{_xy(goal)}</b> &nbsp; tip <b>{_xy(tip)}</b> &nbsp; "
                f"off by <b>{np.linalg.norm(tip - goal[:2]) * 1000:.1f}</b> mm"
                + state["note"]
            )

        _serve(rate, tick)
    except KeyboardInterrupt:
        print("\n[vis] stopping.")
    finally:
        if page is not None:
            page.close()
        arm.stop(wait=False)
        arm.close()


# ----------------------------------------------------------------------
# ex3: joint-space vs workspace trajectories
# ----------------------------------------------------------------------


def _band_pose(minJoint, maxJoint, t1, fraction, t3):
    """Planar angles with theta2 `fraction` of the way across the safe elbow band."""
    return np.array([t1, minJoint[1] + fraction * (maxJoint[1] - minJoint[1]), t3])


def _off_line(point, a, b):
    """Distance from `point` to the segment a-b, all planar."""
    point, a, b = (np.asarray(v, dtype=float) for v in (point, a, b))
    t = np.clip(np.dot(point - a, b - a) / np.dot(b - a, b - a), 0.0, 1.0)
    return float(np.linalg.norm(point - (a + t * (b - a))))


# Fastest any joint is commanded to turn while playing a trajectory, rad/s.
# The simulator's servos clip at 3.14 rad/s and trail well before that, and a
# workspace line through the middle of the arm's reach asks the elbow and wrist
# for several times more at an ordinary playback rate. Past this, a step is
# stretched instead, so the trace shows the trajectory rather than the servos
# falling behind it.
_TRACE_JOINT_SPEED = 1.5


def _play(arm, chain, trajectory, rate, trace_path, color, page, label, line):
    """Servo the arm through `trajectory`'s columns at `rate` per second, tracing the tip.

    A step that would need a joint to move faster than `_TRACE_JOINT_SPEED`
    takes longer instead; the status line says when that happened. The
    status line also keeps the largest distance the traced tip has strayed
    from the straight segment `line`, which is the whole point of comparing
    the two.
    """
    n = trajectory.shape[1]
    qs = [chain.q_from_theta(trajectory[:, col]) for col in range(n)]
    traced = [measured_tip(arm, chain)]  # the arm is parked on column 0
    worst = _off_line(traced[0], *line)
    slowed = False

    def show(col):
        draw_tool(arm.viz, arm, chain)
        draw_points(arm.viz, trace_path, np.array(traced), chain.tool_z,
                    color=color, size=_PATH_POINT_SIZE)
        note = " &nbsp; <i>slowed where a joint would go too fast</i>" if slowed else ""
        page.set_status(f"{label}: point <b>{col + 1}/{n}</b> &nbsp; "
                        f"up to <b>{worst * 1000:.0f}</b> mm off the straight line" + note)

    show(0)
    for col in range(n - 1):
        step = qs[col + 1] - qs[col]
        period = max(1.0 / rate, float(np.max(np.abs(step))) / _TRACE_JOINT_SPEED)
        slowed = slowed or period > 1.0 / rate
        # Commanded at column `col`, moving toward the next one: by the end
        # of the period the setpoint has arrived there, so that is where the
        # tip is read.
        arm.servo_joints(qs[col], velocities=step / period)
        time.sleep(period)
        traced.append(measured_tip(arm, chain))
        worst = max(worst, _off_line(traced[-1], *line))
        show(col + 1)
    # Finish with a planned move onto the last column, not a stop: stopping
    # holds wherever the arm has got to, which can be short of the goal.
    arm.set_joint_targets(qs[-1], speed=_TRACE_JOINT_SPEED)
    traced[-1] = measured_tip(arm, chain)
    worst = max(worst, _off_line(traced[-1], *line))
    show(n - 1)


# The reach the IK descent's fixed step size suits: ex3's own verify arm, three
# 1 m links. See `visualize_ex3`.
_IK_REACH = 3.0  # m


def visualize_ex3(joint_fn, workspace_fn, robot_cls, open_browser=True):
    """Run your two trajectories between the same two poses, one after the other.

    Blue traces the tip under `linear_joint_trajectory`, gold under
    `linear_workspace_trajectory`; the white line is the straight segment
    between the two end points. A joint-space line is a curve in the
    workspace, so only the gold trace should follow the white line. Both
    replay until ctrl-c; the sliders take effect on the next pass.

    `linear_workspace_trajectory` is handed your Robot scaled up to a 3 m
    reach, with the goal scaled to match. Scaling every link of a planar arm
    leaves its joint angles unchanged, so the answer drives the real arm as
    it is. It is scaled because one call of the IK descent (a fixed step,
    100 iterations) was sized for metre-long links and barely converges on
    the xArm's, which would leave each point short of the line.
    """
    arm, chain = start_arm(open_browser=False)
    page = None
    try:
        minJoint, maxJoint = collision_free_limits(arm, chain)
        start = _band_pose(minJoint, maxJoint, -0.9, 0.35, 0.4)
        goal = _band_pose(minJoint, maxJoint, 0.9, 0.65, -0.4)
        start_tip = chain.vertices(start)[3][:2]
        goal_tip = chain.vertices(goal)[3][:2]
        z = chain.tool_z

        page = SliderPage(
            [SliderSpec("points", "points", 5, 150, 1, 40, "", 0),
             SliderSpec("rate", "points / s", 2, 60, 1, 20, "", 0)],
            arm.viz.url,
            title="Joint-space vs workspace trajectories — xArm7",
            open_browser=open_browser,
        )
        scale = _IK_REACH / float(sum(chain.link_lengths))
        robot = robot_cls(np.array(chain.link_lengths) * scale, np.ones(3), np.ones(3), 1.0)
        joint_complaints = Complaints("linear_joint_trajectory", "ex3/linear_joint_trajectory.py")
        ws_complaints = Complaints("linear_workspace_trajectory",
                                   "ex3/linear_workspace_trajectory.py (and ex3/robot.py)")

        draw_polyline(arm.viz, "ex3/line", np.array([[*start_tip, z], [*goal_tip, z]]),
                      color=_LINE_COLOR, width=2.0)
        draw_markers(arm.viz, "ex3/ends", np.array([[*start_tip, z], [*goal_tip, z]]),
                     radius=0.015, color=_LINE_COLOR)
        print("[vis] replaying both trajectories; ctrl-c here to stop.")

        def validated(trajectory, n, name):
            trajectory = _checked(trajectory, "trajectory", (3, n))
            if not np.allclose(trajectory[:, 0], start, atol=1e-6):
                raise ValueError(f"{name}'s first column isn't the start angles")
            end = chain.vertices(trajectory[:, -1])[3][:2]
            if np.linalg.norm(end - goal_tip) > 0.01:
                raise ValueError(f"{name} ends {np.linalg.norm(end - goal_tip) * 1000:.0f} mm "
                                 "from the goal")
            return trajectory

        runs = [
            ("joint space", "ex3/joint", _JOINT_PATH_COLOR, joint_complaints,
             lambda n: joint_fn(start.copy(), goal.copy(), n)),
            ("workspace", "ex3/workspace", _WORKSPACE_PATH_COLOR, ws_complaints,
             lambda n: workspace_fn(robot, start.copy(), goal_tip * scale, n)),
        ]
        while True:
            for label, path, color, complaints, make in runs:
                n = int(page.values["points"])
                rate = float(page.values["rate"])
                arm.set_joint_targets(chain.q_from_theta(start), speed=1.5)
                try:
                    trajectory = validated(make(n), n, label + " trajectory")
                    complaints.clear()
                except Exception as err:
                    complaints(err)
                    page.set_status(f"{label}: <i>not ready yet</i>")
                    time.sleep(1.0)
                    continue
                clear(arm.viz, path)
                _play(arm, chain, trajectory, rate, path, color, page, label,
                      (start_tip, goal_tip))
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[vis] stopping.")
    finally:
        if page is not None:
            page.close()
        arm.stop(wait=False)
        arm.close()


# ----------------------------------------------------------------------
# ex4: gravity compensation, in the vertical plane
# ----------------------------------------------------------------------

# Joint indices (0-based) of the elbow and the wrist the ex4 model covers.
_ELBOW, _WRIST = 3, 5


def _joint_anchor(arm, index):
    """World position of joint `index`'s axis, read under the arm's lock."""
    with arm._lock:
        jid = arm._jid[index]
        body = arm.model.jnt_bodyid[jid]
        R = arm.data.xmat[body].reshape(3, 3)
        return arm.data.xpos[body] + R @ arm.model.jnt_pos[jid]


def _in_plane(origin, angle, offset):
    """`origin` + `offset` rotated by `angle` in the x-z plane (y unchanged)."""
    c, s = math.cos(angle), math.sin(angle)
    dx = c * offset[0] - s * offset[1]
    dz = s * offset[0] + c * offset[1]
    return origin + np.array([dx, 0.0, dz])


def visualize_ex4(module, rate=30.0, open_browser=True):
    """Pose the elbow and wrist with sliders; compare your torques with the arm's.

    The shoulder is upright and everything else but joints 4 and 6 is at zero,
    which is the vertical plane grav_comp.py models. Your
    `compute_gravitational_torques` is evaluated at the angles the arm is
    actually at and set beside `joint_torques`, what a joint torque sensor
    would read. The blue markers are where the model's constants put the two
    centres of mass: they should sit inside the forearm and the wrist.
    """
    for name in ("ELBOW_OFFSET", "WRIST_OFFSET", "LINK_A_LENGTH", "COM_A", "COM_B",
                 "planar_angles", "to_joint_torques", "compute_gravitational_torques"):
        if not hasattr(module, name):
            sys.exit(f"ex4/grav_comp.py has no {name}; it is part of the provided code.")
    tolerance = float(getattr(module, "TORQUE_TOLERANCE", 0.1))

    home = np.zeros(7)
    home[_ELBOW] = -module.ELBOW_OFFSET  # forearm level

    arm = SimulatedXArm7(visualize=False, safety_box=None, guard=True)
    page = None
    try:
        arm.viz = MeshcatVisualizer(arm.model, arm.data, open_browser=False)
        print(f"[vis] meshcat: {arm.viz.url}")
        arm.set_joint_targets(home, speed=0.8)

        # Slider ranges in the model's angles, from the two joints' own limits.
        q4_low, q4_high = arm.position_limit[_ELBOW]
        q6_low, q6_high = arm.position_limit[_WRIST]
        page = SliderPage(
            [
                SliderSpec("a", "theta_a", module.ELBOW_OFFSET + q4_low,
                           module.ELBOW_OFFSET + q4_high, 0.005, 0.0, " rad", 2),
                SliderSpec("b", "theta_b", module.WRIST_OFFSET - q6_high,
                           module.WRIST_OFFSET - q6_low, 0.005,
                           module.WRIST_OFFSET - math.radians(30), " rad", 2),
            ],
            arm.viz.url,
            title="Gravity compensation — xArm7",
            open_browser=open_browser,
        )
        complaints = Complaints("compute_gravitational_torques", "ex4/grav_comp.py")
        print("[vis] drag the sliders; ctrl-c here to stop.")

        def tick():
            values = page.values
            q = home.copy()
            q[_ELBOW] = values["a"] - module.ELBOW_OFFSET
            q[_WRIST] = module.WRIST_OFFSET - values["b"]
            q = np.clip(q, arm.position_limit[:, 0], arm.position_limit[:, 1])
            blocked = arm.check_safety(q) is not None
            if not blocked:
                arm.servo_joints(q)

            theta_a, theta_b = module.planar_angles(arm.joint_values)
            elbow = _joint_anchor(arm, _ELBOW)
            wrist = _in_plane(elbow, theta_a, (module.LINK_A_LENGTH, 0.0))
            com_a = _in_plane(elbow, theta_a, module.COM_A)
            com_b = _in_plane(wrist, theta_a + theta_b, module.COM_B)
            draw_polyline(arm.viz, "ex4/model", np.array([elbow, wrist, com_b]),
                          color=_COM_COLOR, width=3.0)
            draw_markers(arm.viz, "ex4/coms", np.array([com_a, com_b]),
                         radius=0.025, color=_COM_COLOR)

            actual = arm.joint_torques[[_ELBOW, _WRIST]]
            settling = float(np.max(np.abs(arm.joint_velocities))) > 0.02
            try:
                yours = _checked(
                    module.to_joint_torques(module.compute_gravitational_torques(theta_a, theta_b)),
                    "torques", (2,),
                )
                complaints.clear()
            except Exception as err:
                complaints(err)
                yours = None

            cells = []
            for k, joint in enumerate((4, 6)):
                if yours is None:
                    verdict = "—"
                elif settling:
                    verdict = "settling…"
                else:
                    verdict = "PASS" if abs(yours[k] - actual[k]) <= tolerance else "FAIL"
                mine = "—" if yours is None else f"{yours[k]:+.3f}"
                cells.append(f"joint {joint}: yours <b>{mine}</b> arm <b>{actual[k]:+.3f}</b> "
                             f"N&middot;m <b>{verdict}</b>")
            note = " &nbsp; <i>that pose self-collides; holding</i>" if blocked else ""
            page.set_status(" &nbsp;&nbsp; ".join(cells) + note)

        _serve(rate, tick)
    except KeyboardInterrupt:
        print("\n[vis] stopping.")
    finally:
        if page is not None:
            page.close()
        arm.stop(wait=False)
        arm.close()
