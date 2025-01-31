#!/usr/bin/env python3

import torch
import sys

def load_csv_as_list_of_lists(filepath: str):
    lines = []
    with open(filepath, 'r') as f:
        for row in f:
            row = row.strip()
            if row:
                floats = list(map(float, row.split(',')))
                lines.append(floats)
    return lines

def reorder_lines_python(lines, num_env, batch_size):
    """
    The 'lines' are currently in shape => (num_env x batch_size).
    i.e. lines = [
      # environment=0, step=0
      # environment=0, step=1
      ...
      # environment=0, step= (batch_size-1)
      # environment=1, step=0
      ...
      # environment= (num_env-1), step= (batch_size-1)
    ]

    But we want them in shape => (batch_size x num_env).
    i.e. the c++ approach is (step=0, env=0), (step=0, env=1), ...
    So we reorder them so line i => lines2[i],
    lines2[ step * num_env + env ] = lines[ env * batch_size + step ].

    We do:
       new_lines[ i ] = lines[ old_index ]
    Where i => range(num_env*batch_size).

    Return the new reorder lines.
    """
    # safety check
    if len(lines) != num_env * batch_size:
        print(f"Mismatch: expected {num_env} * {batch_size} lines, found {len(lines)}.")
        return lines  # can't fix

    new_lines = [None]*(num_env*batch_size)

    # for env in [0..num_env), for step in [0..batch_size):
    #  old_idx = env*batch_size + step
    #  new_idx = step*num_env + env
    # Then new_lines[new_idx] = lines[old_idx]
    idx = 0
    for env in range(num_env):
        for step in range(batch_size):
            old_idx = env*batch_size + step
            new_idx = step*num_env + env
            new_lines[new_idx] = lines[old_idx]
    return new_lines

def compare_tensors(tA, tB, rtol=1e-5, atol=1e-6):
    """
    Simple compare with torch.allclose
    """
    if tA.shape != tB.shape:
        print("Shape mismatch:", tA.shape, tB.shape)
        return False

    return torch.allclose(tA, tB, rtol=rtol, atol=atol)

def main():
    python_csv = "PythonTransitions.csv"
    unreal_csv = "UnrealTransitions.csv"


    # 1) Load CSV lines
    lines_py = load_csv_as_list_of_lists(python_csv)
    lines_un = load_csv_as_list_of_lists(unreal_csv)

    # 3) Convert to PyTorch
    t_py = torch.tensor(lines_py, dtype=torch.float32)
    t_un = torch.tensor(lines_un, dtype=torch.float32)

    # 4) Compare
    if compare_tensors(t_py, t_un):
        print("SUCCESS: CSV lines match after reordering.")
    else:
        diff = (t_py - t_un).abs()
        max_diff = diff.max().item()
        locs = (diff == diff.max()).nonzero(as_tuple=False)
        print(f"FAIL: Mismatch. max_diff={max_diff:.6f}, sample location(s): {locs[:5]}")

if __name__ == "__main__":
    main()
