import argparse
import matplotlib.pyplot as plt
import numpy as np
import random
import time

import test_setup as setup
import refit

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
  start_time = time.time()
  if strategy == "anneal":
    curves = refit.anneal_close_curves(curves, distance_threshold)
  elif strategy == "polyfit":
    curves = refit.polyfit_merge_curves(curves)
  else:
    raise ValueError(f"Invalid refit strategy: {strategy}")
  print(f"Refit strategy took {time.time() - start_time:.2f} seconds")
  print(f"Curve statistics after applying {strategy} refit strategy:")
  print_curve_stats(curves)

  # plot curves
  for curve in curves:
    plt.plot([state[1] for state in curve], [state[2] for state in curve], 'b,')
  plt.show()

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

if __name__ == "__main__":
  main()