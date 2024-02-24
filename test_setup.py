import numpy as np
import random

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
  init_vel = lambda: polar_to_cartesian([random.uniform(min_vel, max_vel), random.uniform(0, 2 * np.pi)])

  curves = [integrate_euler(init_pos(), init_vel(), brownian_acc(brownian_sigma), dt, max_time)
            for _ in np.arange(n_curves)]
  
  return id_curves(curves)

def brownian_acc(sigma=1.0):
  """Simulate random (Brownian) motion."""
  return lambda: np.array([random.gauss(0, sigma) for _ in range(2)])

def polar_to_cartesian(polar):
  """Convert polar coordinates to Cartesian coordinates."""
  return np.array([polar[0] * np.cos(polar[1]), polar[0] * np.sin(polar[1])])