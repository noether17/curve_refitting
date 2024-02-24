from itertools import combinations
import numpy as np
from scipy.spatial.distance import mahalanobis

def polar_to_cartesian(polar):
  """Convert polar coordinates to Cartesian coordinates."""
  return np.array([polar[0] * np.cos(polar[1]), polar[0] * np.sin(polar[1])])

def polyfit_curve(curve, degree=2):
  times = np.array([state[0] for state in curve])
  x_values = np.array([state[1] for state in curve])
  y_values = np.array([state[2] for state in curve])
  return np.polyfit(times, np.column_stack((x_values, y_values)), degree, cov=True)

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