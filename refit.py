import numpy as np
import random
import util

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
    new_curve1 = curve1.copy()
    new_curve2 = curve2.copy()
    move_index = random.randint(0, len(curve1) + len(curve2)- 1)
    if move_index < len(curve1):
      curve1_index = move_index
      curve2_index = next((k for k, state in enumerate(curve2) if state[0] == curve1[curve1_index][0]), None)
    else:
      curve1_index = next((k for k, state in enumerate(curve1) if state[0] == curve2[move_index - len(curve1)][0]), None)
      curve2_index = move_index - len(curve1)
    if curve1_index is not None and curve2_index is not None:
      new_curve1[curve1_index], new_curve2[curve2_index] = new_curve2[curve2_index], new_curve1[curve1_index]
    elif curve1_index is not None:
      new_curve2.append(curve1[curve1_index])
      new_curve1.pop(curve1_index)
    elif curve2_index is not None:
      new_curve1.append(curve2[curve2_index])
      new_curve2.pop(curve2_index)
    else:
      raise ValueError("Invalid move index")

    # accept or reject the new state
    new_E = energy(new_curve1) + energy(new_curve2)
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