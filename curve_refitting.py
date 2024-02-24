import argparse
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.spatial.distance import mahalanobis

import test_setup as setup
import util

def main():
  # parse command-line arguments
  parser = argparse.ArgumentParser(description="Curve refitting")
  parser.add_argument("-n", "--n_curves", type=int, default=1000, help="Number of curves to simulate")
  parser.add_argument("-p", "--n_points", type=int, default=25, help="Number of points per curve")
  parser.add_argument("-b", "--brownian_parameter", type=float, default=1.0e-2, help="Brownian parameter")
  parser.add_argument("-s", "--scenario", type=str, default="bisect", help="Scenario: bisect, unzip, scramble")
  parser.add_argument("-g", "--gap_portion", type=float, default=0.1, help="Portion of curve to remove (for bisect scenario)")
  parser.add_argument("-d", "--distance_threshold", type=float, default=2.0e2, help="Distance threshold (for scramble scenario)")
  parser.add_argument("-r", "--refit_strategy", type=str, default="anneal", help="Refit strategy: anneal, polyfit")
  args = parser.parse_args()
  n_curves = args.n_curves
  n_points = args.n_points
  brownian_parameter = args.brownian_parameter
  scenario = args.scenario
  gap_portion = args.gap_portion
  distance_threshold = args.distance_threshold
  strategy = args.refit_strategy

  # set random seed for reproducibility
  random.seed(0)

  # simulate trajectories
  curves = setup.generate_curves(n_curves, n_points, brownian_parameter)

  # apply scenario
  if scenario == "bisect":
    curves = setup.bisect_curves(curves, gap_portion)
  elif scenario == "unzip":
    curves = setup.unzip_curves(curves)
  elif scenario == "scramble":
    curves = setup.scramble_close_curves(curves, distance_threshold)
  else:
    raise ValueError(f"Invalid scenario: {scenario}")
  print(f"Curve statistics after applying {scenario} scenario:")
  print_curve_stats(curves)

  # apply refit strategy
  if strategy == "anneal":
    curves = anneal_close_curves(curves, distance_threshold)
  elif strategy == "polyfit":
    curves = polyfit_merge_curves(curves)
  else:
    raise ValueError(f"Invalid refit strategy: {strategy}")
  print(f"Curve statistics after applying {strategy} refit strategy:")
  print_curve_stats(curves)

  # plot curves
  for curve in curves:
    plt.plot([state[1] for state in curve], [state[2] for state in curve], 'b,')
  plt.show()

def polyfit_merge_curves(curves):
  result = []
  used_indices = set()
  for pair, distance in util.find_close_curves(curves, 1.0e2):
    if pair[0] not in used_indices and pair[1] not in used_indices:
      result.append(curves[pair[0]] + curves[pair[1]])
      used_indices.add(pair[0])
      used_indices.add(pair[1])
  for i in range(len(curves)):
    if i not in used_indices:
      result.append(curves[i])

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

def anneal_close_curves(curves, distance_threshold):
  pairs = util.find_close_curves(curves, distance_threshold)
  for pair, distance in pairs:
    curves = anneal_curves(curves, pair)
  return [curve for curve in curves if len(curve) > 0]

if __name__ == "__main__":
  main()