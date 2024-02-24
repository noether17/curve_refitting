import numpy as np

def integrate_euler(pos, vel, acc, dt, max_time):
  t = 0.0
  trajectory = [np.hstack([t, pos])]
  for t in np.arange(dt, max_time, dt):
    pos += vel * dt
    vel += acc() * dt
    trajectory.append(np.hstack([t, pos]))
  return trajectory