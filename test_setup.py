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

def brownian_acc(sigma=1.0):
  """Simulate random (Brownian) motion."""
  return lambda: np.array([random.gauss(0, sigma) for _ in range(2)])

def polar_to_cartesian(polar):
  """Convert polar coordinates to Cartesian coordinates."""
  return np.array([polar[0] * np.cos(polar[1]), polar[0] * np.sin(polar[1])])