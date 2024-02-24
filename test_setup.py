import numpy as np
import random

import util

def integrate_euler(pos, vel, acc, dt, max_time):
  """Use Euler's method to generate curves given acceleration function."""
  t = 0.0
  trajectory = [np.hstack([t, pos])]
  for t in np.arange(dt, max_time, dt):
    pos += vel * dt
    vel += acc() * dt
    trajectory.append(np.hstack([t, pos]))
  return trajectory

def id_curves(curves):
  """Tag individual states with curve id for validation."""
  return [[np.hstack([state, i]) for state in curve] for i, curve in enumerate(curves)] # id is last element

def generate_curves(n_curves, n_points, brownian_parameter):
  """Generate random curves based on Brownian motion."""
  frame_width = 1.0
  max_time = n_points
  dt = 1.0
  max_vel = frame_width / max_time / 2.0
  min_vel = max_vel / 2.0
  brownian_sigma = max_vel * brownian_parameter

  init_pos = lambda: np.array([random.uniform(-frame_width / 2.0, frame_width / 2.0) for _ in range(2)])
  init_vel = lambda: util.polar_to_cartesian([random.uniform(min_vel, max_vel), random.uniform(0, 2 * np.pi)])

  curves = [integrate_euler(init_pos(), init_vel(), brownian_acc(brownian_sigma), dt, max_time)
            for _ in np.arange(n_curves)]
  
  return id_curves(curves)

def brownian_acc(sigma=1.0):
  """Simulate random (Brownian) motion."""
  return lambda: np.array([random.gauss(0, sigma) for _ in range(2)])

def bisect_curves(curves, gap_portion):
  """Bisect the curves, removing a portion of the curve."""
  result = []
  for curve in curves:
    gap_size = int(len(curve) * gap_portion)
    gap_index = int((len(curve) - gap_size) / 2)
    result.append(curve[:gap_index])
    result.append(curve[gap_index + gap_size:])
  return result

def unzip_curves(curves):
  """Split a curve into two separate curves, randomly distributing the points."""
  result = []
  for curve in curves:
    assignments = [random.randint(0, 1) for _ in np.arange(len(curve))]
    result.append([curve[i] for i in np.arange(len(curve)) if assignments[i] == 0])
    result.append([curve[i] for i in np.arange(len(curve)) if assignments[i] == 1])
  return result

def scramble_curves(curves, pairs):
  for pair, distance in pairs:
    curve1 = curves[pair[0]]
    curve2 = curves[pair[1]]
    combined_curve = curve1 + curve2
    random.shuffle(combined_curve)
    combined_curve = sorted(combined_curve, key=lambda x: x[0]) # make sure curves do not duplicate times
    curves[pair[0]] = combined_curve[0::2]
    curves[pair[1]] = combined_curve[1::2]
  return curves

def scramble_close_curves(curves, distance_threshold):
  """Conflates nearby curves by randomly redistributing their points."""
  return scramble_curves(curves, util.find_close_curves(curves, distance_threshold))