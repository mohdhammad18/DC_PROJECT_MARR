# Robot Localization Simulator

An interactive, modular simulator that shows how **wheel odometry**, an **IMU (gyroscope)** and an
intermittent **GPS** can be fused with an **Extended Kalman Filter (EKF)** to localize a
differential-drive robot.

> **Scope:** this first version is **planar 2D** (`z = roll = pitch = 0`). It estimates `[x, y, ψ]` only
> and does **not** implement full 6-DoF localization.

The simulator keeps three trajectories on one time axis:

| Trajectory | What it is |
|---|---|
| **Ground truth** | The actual simulated motion of the robot. |
| **Odometry-only** | Dead reckoning from noisy wheel odometry (+ gyro yaw rate). Never uses GPS, so it drifts. |
| **EKF fused** | The same prediction, corrected by noisy, intermittent GPS fixes through the EKF. |

## Main features

- Differential-drive kinematics with exact-arc ground-truth integration and consistent angle wrapping.
- Explicit local ↔ global coordinate-frame transformations with a visual explanation.
- Configurable sensors with reproducible noise: wheel encoders (noise + scale error), gyro (noise, bias,
  slow bias drift, optional noisy heading), GPS (noise, update rate, random dropouts, can be disabled).
- A hand-written EKF (NumPy only) with Joseph-form covariance update, wrapped yaw innovations and every
  intermediate quantity exposed (predicted state/covariance, innovation, `S`, `K`, corrected state, updated `P`).
- Predefined paths: straight line, square, circle, figure eight, custom segment sequence.
- Six ready-to-run scenarios that can be compared side by side.
- Streamlit dashboard: overview, trajectory comparison, error analysis (mean / RMSE / max / final / yaw),
  EKF internals, coordinate frames, sensor data, scenario comparison and an animated replay.
- Simulation controls that work: **Start**, **Pause/Resume**, **Reset**, **Step**, **Run complete simulation**.
- 100+ unit and app-level tests.

## Installation

Requires **Python 3.10 or newer**. No dependencies are assumed to be installed.

**Windows**

```powershell
git clone https://github.com/mohdhammad18/DC_PROJECT_MARR.git
cd DC_PROJECT_MARR

python -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt

streamlit run app.py
```

**Linux / macOS**

```bash
git clone https://github.com/mohdhammad18/DC_PROJECT_MARR.git
cd DC_PROJECT_MARR

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

streamlit run app.py
```

Streamlit prints a local URL (normally <http://localhost:8501>). Always run the commands from the repository
root. No environment variables or secrets are required, so there is no `.env` file.

### Running the tests

```bash
pytest -q
```

This runs the unit tests and a headless smoke test of the Streamlit app (about one to two minutes).

## Using the dashboard

1. The first load runs the default configuration so you immediately see real simulation data.
2. Change settings in the **sidebar**: robot geometry, trajectory, time step and duration, sensor noise,
   gyro bias, GPS rate / noise / dropouts, random seed, EKF tuning and the environment boundary.
   Or pick a **scenario preset** and press **Apply preset**.
3. Settings only take effect when you press **Start**, **Reset** or **Run complete simulation**
   (a banner tells you when the sidebar differs from the simulation shown, so results are never silently stale).
4. Use the views below the metrics to explore the results.

| Control | Behaviour |
|---|---|
| Start | Builds a fresh simulation from the sidebar settings and plays it from `t = 0`. |
| Pause / Resume | Stops / continues the playback. |
| Reset | Returns to `t = 0` (nothing is run). |
| Step | Advances exactly one time step. |
| Run complete simulation | Rebuilds and runs to the end immediately. |

The **Animated replay** view is a client-side Plotly animation of the simulation data produced so far.

### Scenarios

| # | Scenario | What it demonstrates |
|---|---|---|
| 1 | Ideal sensors | No noise/bias/scale error → both estimates match the truth (second-order motion model). |
| 2 | Noisy odometry | Wheel-only dead reckoning with noisy, mis-scaled encoders → drift. |
| 3 | Noisy GPS | Perfect odometry/gyro, poor GPS (1.5 m, 30 % lost) → the EKF filters the scattered fixes. |
| 4 | Odometry + IMU | Same wheels as #2, plus gyro yaw rate → heading-assisted dead reckoning, still no GPS. |
| 5 | Full sensor fusion | Odometry + gyro + GPS through the EKF → drift is corrected. |
| 6 | GPS-denied | Scenario 5 without GPS → the EKF can only predict and drifts like dead reckoning. |

Scenarios share the trajectory, duration, time step and seed from the sidebar, so they are directly comparable
in the **Scenario comparison** view (metrics table, overlaid trajectories and error curves).

## Architecture

```
app.py                      Streamlit entry point (wiring + session state only)
requirements.txt            Direct dependencies with version constraints
pytest.ini                  Test configuration (puts the repo root on the import path)
.streamlit/config.toml      Streamlit settings (usage statistics off)
config/default_config.yaml  Default simulation settings

src/                        Simulation core - no Streamlit / Plotly imports
    coordinate_frames.py    wrap_angle, R(psi), local<->global transforms (points and free vectors)
    robot.py                Differential-drive kinematics and ground-truth robot
    odometry.py             Motion model and odometry-only dead reckoning
    sensors.py              Wheel odometry, IMU, GPS models and seeded random streams
    ekf.py                  Extended Kalman Filter
    trajectories.py         Command generators: straight, square, circle, figure eight, custom
    config.py               Typed, validated configuration; YAML loading
    simulation.py           Engine: truth + sensors + odometry + EKF on one time axis
    metrics.py              Position / yaw error metrics
    scenarios.py            The six preset scenarios

visualization/              Plotly figures and Streamlit panels - no simulation logic
    trajectory.py           Trajectory comparison and coordinate-frame figures
    robot_animation.py      Animated replay
    sensor_plots.py         Error, covariance, sensor, GPS and scenario figures
    dashboard.py            Streamlit panels for each dashboard section
    controls.py             Sidebar widgets and session-state handling

tests/                      pytest suite (robot, frames, EKF, simulation/sensors/config/metrics, app)
```

## Mathematical model

### Differential-drive kinematics

With wheel radius `r`, wheel separation `b` and wheel angular velocities `α_L`, `α_R`:

```
v     = (r / 2) · (α_R + α_L)
ω     = (r / b) · (α_R − α_L)
```

The ground truth integrates a constant `(v, ω)` over each step along its exact circular arc
(straight line when `ω → 0`). Headings are wrapped to `[−π, π)`.

### Coordinate frames

* **Global frame (GCSF)**: `X_G`, `Y_G` fixed to the world.
* **Local frame (LCSF)**: `X_L` points along the heading `ψ`, `Y_L` to the robot's left; its origin is the robot.

```
R(ψ) = [[cos ψ, −sin ψ],
        [sin ψ,  cos ψ]]

p_G = p_robot_G + R(ψ) · p_L          (local point → global point)
p_L = R(ψ)ᵀ · (p_G − p_robot_G)       (global point → local point)
v_G = R(ψ) · v_L                      (free vectors: rotation only)
```

Wheel measurements live in the local frame, GPS fixes live in the global frame, and the EKF state is global.
`coordinate_frames.py` offers separate point and free-vector transforms so the two are never mixed up.
The *Coordinate frames* view draws both frames and maps a chosen local vector into the global frame.

### EKF

State `X = [x, y, ψ]ᵀ`, control inputs `v` (wheel odometry) and `ω` (gyro, or wheel-derived when the IMU is off).

**Prediction**

```
x(k+1)   = x(k) + v cos ψ dt
y(k+1)   = y(k) + v sin ψ dt
ψ(k+1)   = ψ(k) + ω dt

F = [[1, 0, −v sin ψ dt],
     [0, 1,  v cos ψ dt],
     [0, 0,  1         ]]

P⁻ = F P Fᵀ + Q ,      Q = diag(q_pos², q_pos², q_yaw²) · dt
```

`ekf.integration: midpoint` evaluates the heading at `ψ + ω dt / 2` (second-order accurate) and adjusts `F` accordingly;
the default `euler` is the model above.

**GPS update** (global position, `z = [x_gps, y_gps]ᵀ`, `H = [[1,0,0],[0,1,0]]`)

```
r = z − H x⁻                      innovation
S = H P⁻ Hᵀ + R                   innovation covariance
K = P⁻ Hᵀ S⁻¹                     Kalman gain (computed with a linear solve, not an explicit inverse)
x̂ = x⁻ + K r
P = (I − K H) P⁻ (I − K H)ᵀ + K R Kᵀ       Joseph form
```

The Joseph form equals `(I − K H) P⁻` for the optimal gain but stays symmetric and positive semidefinite
under rounding. An ill-conditioned `S` raises `NumericalError`; the engine records the rejected update
in `gps_status` and `warnings` (and logs it) instead of ignoring it.

**Optional heading update**: with `H = [0, 0, 1]` the innovation is wrapped, `r = wrap(ψ_meas − ψ⁻)`.

### Sensor models

| Sensor | Model |
|---|---|
| Wheel odometry | `α_meas = α_true · (1 + scale_error)` (right wheel) `+ N(0, σ²)` on each wheel; `v, ω` follow from the kinematics above. |
| IMU gyro | `ω_gyro = ω_true + bias_k + N(0, σ²)`, `bias_{k+1} = bias_k + N(0, σ_rw² dt)`. |
| IMU heading (optional) | `ψ_meas = wrap(ψ_true + N(0, σ²))` at `heading_rate_hz`. |
| GPS | Global `[x, y] + N(0, σ²I)` every `1/rate_hz`; each scheduled fix is lost with `dropout_probability`; can be disabled. |

The odometry-only path and the EKF prediction use the **same** measured `(v, ω)`; the only difference is the
GPS (and optional heading) correction. Each sensor draws from its own random stream derived from the master
seed, so a seed reproduces a run exactly and changing one sensor never alters another's noise.

## Configuration

All defaults live in [`config/default_config.yaml`](config/default_config.yaml), organised in the sections
`robot`, `trajectory`, `simulation`, `odometry`, `imu`, `gps` and `ekf`. Every value is editable in the sidebar
and the current setup can be downloaded as YAML. Unknown keys and invalid values (non-positive wheel radius or
`dt`, negative noise, bad probabilities, …) are rejected with a message listing every problem.

Use the engine without the UI:

```python
from src.config import load_config
from src.simulation import run_simulation
from src.metrics import compare_estimators

cfg = load_config()                      # or load_config("my_config.yaml")
cfg.trajectory.name = "figure_eight"
sim = run_simulation(cfg)
print(compare_estimators(sim.to_dataframe()))
```

Notes on units and tuning:

* `q_pos` / `q_yaw` are process-noise densities (`m/√s`, `rad/√s`); `Q` scales with `dt`.
* With `ekf.match_sensor_noise: true`, `R` is built from the GPS / heading noise settings (floored at
  0.01 m / 1 mrad so it stays positive definite). Switch it off to study a mismatched `R`.
* Custom trajectories are lists of `{duration, v, omega}` segments; the robot stops after the last one.
* `duration / dt` is limited to 200 000 steps.

## Known limitations

- Planar 2D only; no roll/pitch, no 6-DoF localization.
- The ground truth is noise-free kinematics (no wheel slip, no actuator dynamics, no obstacles or collisions);
  the optional boundary is only drawn and reported, not enforced.
- The EKF state has three elements; gyro bias is **not** estimated (it shows up as heading drift that GPS
  corrects), and heading is only observable through its correlation with position.
- `Q` is a simple diagonal matrix, not derived from the sensor noise through the control Jacobian.
- GPS outliers are not gated (only numerically invalid updates are rejected).
- GPS timing is quantised to whole time steps.
- Playback re-renders the page every few steps, so very long runs play slowly; use *Run complete simulation*
  or the client-side *Animated replay* instead.

## Future extensions

- Estimate gyro bias in the EKF state; derive `Q` from sensor noise via the control Jacobian.
- Innovation gating / robust GPS outlier rejection and NEES/NIS consistency plots.
- Wheel slip and terrain models for the ground truth; obstacle-aware environments.
- Full 6-DoF (3D) state estimation with roll/pitch.
- Additional sensors and estimators (LiDAR, visual SLAM, UKF, particle filter) - intentionally out of scope for this version.
