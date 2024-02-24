import argparse
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.spatial.distance import mahalanobis

def main():
  # simulation parameters
  n_curves = 1000
  max_time = 25.0
  dt = 1.0
  frame_width = 1.0
  max_vel = frame_width / max_time / 2.0
  min_vel = max_vel / 2.0
  brownian_sigma = max_vel / 1.0e2
  gap_portion = 0.1

  # set random seed for reproducibility
  random.seed(0)

  # initialize pos and vel functions
  init_pos = lambda: np.array([random.uniform(0.0, frame_width), random.uniform(0.0, frame_width)])
  init_vel = lambda: polar_to_cartesian([random.uniform(min_vel, max_vel), random.uniform(0, 2 * np.pi)])

  # simulate trajectories
  curves = [integrate_euler(init_pos(), init_vel(), brownian_acc(brownian_sigma), dt, max_time) for _ in np.arange(n_curves)]

  # tag individual states with curve id for validation
  curves = id_curves(curves)

  # split the curves
  #curves = split_curves(curves, gap_portion)

  # merge the curves
  #curves = merge_curves(curves)

  # print curve stats
  print_curve_stats(curves)

  curves = unzip_curves(curves)

  # find close curves
  close_curves = find_close_curves(curves, 2.0e2)
  #plot_pairs(curves, close_curves)

  # scramble the curves
  #curves = scramble_curves(curves, close_curves)
  #plot_pairs(curves, close_curves)

  # print curve stats, post-scramble
  print_curve_stats(curves)

  # anneal the curves
  #plot_pairs(curves, close_curves)
  curves = unscramble_curves(curves, close_curves)
  #plot_pairs(curves, close_curves)

  # print curve stats, post-unscramble
  print_curve_stats(curves)

  # plot curves
  plot_min = -frame_width / 2.0 # add some padding
  plot_max = 3.0*frame_width / 2.0
  for curve in curves:
    plt.plot([state[1] for state in curve], [state[2] for state in curve], 'b,')
  plt.xlim(plot_min, plot_max) # add some padding
  plt.ylim(plot_min, plot_max)
  plt.show()

def integrate_euler(pos, vel, acc, dt, max_time):
  t = 0.0
  trajectory = [np.hstack([t, pos])]
  for t in np.arange(dt, max_time, dt):
    pos += vel * dt
    vel += acc() * dt
    trajectory.append(np.hstack([t, pos]))
  return trajectory

def brownian_acc(sigma=1.0):
  return lambda: np.array([random.gauss(0, sigma) for _ in range(2)])

def polar_to_cartesian(polar):
  return np.array([polar[0] * np.cos(polar[1]), polar[0] * np.sin(polar[1])])

def id_curves(curves):
  return [[np.hstack([state, i]) for state in curve] for i, curve in enumerate(curves)] # id is last element

def split_curves(curves, gap_portion):
  result = []
  for curve in curves:
    gap_size = int(len(curve) * gap_portion)
    gap_index = int((len(curve) - gap_size) / 2)
    result.append(curve[:gap_index])
    result.append(curve[gap_index + gap_size:])
  return result

def unzip_curves(curves):
  result = []
  for curve in curves:
    assignments = [random.randint(0, 1) for _ in np.arange(len(curve))]
    result.append([curve[i] for i in np.arange(len(curve)) if assignments[i] == 0])
    result.append([curve[i] for i in np.arange(len(curve)) if assignments[i] == 1])
  return result

def polyfit_curve(curve, degree=2):
  times = np.array([state[0] for state in curve])
  x_values = np.array([state[1] for state in curve])
  y_values = np.array([state[2] for state in curve])
  return np.polyfit(times, np.column_stack((x_values, y_values)), degree, cov=True)

def merge_curves(curves):
  result = []
  used_indices = set()
  for pair, distance in find_close_curves(curves, 1.0e2):
    if pair[0] not in used_indices and pair[1] not in used_indices:
      result.append(curves[pair[0]] + curves[pair[1]])
      used_indices.add(pair[0])
      used_indices.add(pair[1])
  for i in range(len(curves)):
    if i not in used_indices:
      result.append(curves[i])

  return result

def find_close_curves(curves, distance_threshold):
  fit_params = [list(polyfit_curve(curve)) for curve in curves]
  result = []
  for i, j in combinations(range(len(curves)), 2):
    combined_cov = (fit_params[i][1] + fit_params[j][1]) / 2.0
    x_distance = mahalanobis(fit_params[i][0][:, 0], fit_params[j][0][:, 0], np.linalg.inv(combined_cov[:, :, 0]))
    if x_distance > distance_threshold: continue # no need to compute y distance
    y_distance = mahalanobis(fit_params[i][0][:, 1], fit_params[j][0][:, 1], np.linalg.inv(combined_cov[:, :, 1]))
    distance = np.sqrt(x_distance**2 + y_distance**2)
    if distance < distance_threshold:
      result.append([(i, j), distance])
  result = sorted(result, key=lambda x: x[1])
  return result

def print_curve_stats(curves):
  ids = np.unique([state[3] for curve in curves for state in curve])
  print(f"Number of original curves: {ids.size}")
  print(f"Number of observed curves: {len(curves)}")

  n_mixed = len([curve for curve in curves if len(np.unique([state[3] for state in curve])) > 1])
  print(f"Number of mixed curves: {n_mixed}")

  n_split = len([id for id in ids if len([curve for curve in curves if id in [state[3] for state in curve]]) > 1])
  print(f"Number of split curves: {n_split}")

def plot_pairs(curves, pairs):
  print(f"Number of pairs: {len(pairs)}")
  for pair, distance in pairs:
    plt.plot([state[1] for state in curves[pair[0]]], [state[2] for state in curves[pair[0]]], 'b.', label=f"Curve {pair[0]}")
    plt.plot([state[1] for state in curves[pair[1]]], [state[2] for state in curves[pair[1]]], 'r.', label=f"Curve {pair[1]}")
    plt.title(f"Distance: {distance}")
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.show()

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

def energy(curve):
  if len(curve) == 0: return 0.0
  if len(curve) < 3: return 0.0
  times = np.array([state[0] for state in curve])
  x_values = np.array([state[1] for state in curve])
  y_values = np.array([state[2] for state in curve])
  p, residuals, rank, singular_values, rcond = np.polyfit(times, np.column_stack((x_values, y_values)), 2, full=True)
  # simple sum of squared residuals will likely fail to combine curves; may need a softening constant
  return np.sum(residuals)

def anneal_curves(curves, pair):
  curve1 = curves[pair[0]]
  curve2 = curves[pair[1]]
  E = energy(curve1) + energy(curve2)
  T_max = 10.0 # initial temperature
  T_min = 1.0e-3*E # final temperature
  tau = 1.0e3 # cooling timescale
  T = T_max
  t = 0.0
  while T > T_min:
    # cooling
    t += 1
    T = T_max * np.exp(-t / tau)

    # propose a new state
    if len(curve1) == 0: break
    new_curve1 = curve1.copy()
    new_curve2 = curve2.copy()
    curve1_index = random.randint(0, len(curve1) - 1)
    curve2_index = next((k for k, state in enumerate(curve2) if state[0] == curve1[curve1_index][0]), None)
    if curve2_index is not None:
      new_curve1[curve1_index], new_curve2[curve2_index] = new_curve2[curve2_index], new_curve1[curve1_index]
    else:
      new_curve2.append(curve1[curve1_index])
      new_curve1.pop(curve1_index)
    new_E = energy(new_curve1) + energy(new_curve2)

    # accept or reject the new state
    if new_E < E or random.random() < np.exp(-(new_E - E) / T):
      curve1 = new_curve1
      curve2 = new_curve2
      E = new_E
  curves[pair[0]] = curve1
  curves[pair[1]] = curve2
  return curves

def unscramble_curves(curves, pairs):
  for pair, distance in pairs:
    curves = anneal_curves(curves, pair)
  return [curve for curve in curves if len(curve) > 0]

if __name__ == "__main__":
  main()