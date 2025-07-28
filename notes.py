import numpy as np

x = np.array([-10, 1, 5, 12])
bins = np.array([0, 2, 4, 6, 8, 10])
binned = np.digitize(x, bins)
print(binned)

x = np.array([0, 1, 2, 3])
np.add.at(x, [0], 1)
print(x)

x = np.zeros(5)
print(x)

x = np.full(5, 5)
print(x)

x = np.ones((2, 3))
print(x.shape[1])

x = np.zeros((2, 3))
print(f"numpy zeroes: {x}")

x = np.array([10, 10, 10])
print(x.shape)


def quick_sort(arr, simulation=False):
    """Quick sort
    Complexity: best O(n log(n)) avg O(n log(n)), worst O(N^2)
    """

    iteration = 0
    if simulation:
        print("iteration", iteration, ":", *arr)
    arr, _ = quick_sort_recur(arr, 0, len(arr) - 1, iteration, simulation)
    return arr


def quick_sort_recur(arr, first, last, iteration, simulation):
    if first < last:
        pos = partition(arr, first, last)
        # Start our two recursive calls
        if simulation:
            iteration = iteration + 1
            print("iteration", iteration, ":", *arr)

        _, iteration = quick_sort_recur(arr, first, pos - 1, iteration, simulation)
        _, iteration = quick_sort_recur(arr, pos + 1, last, iteration, simulation)

    return arr, iteration


def partition(arr, first, last):
    wall = first
    for pos in range(first, last):
        if arr[pos] < arr[last]:  # last is the pivot
            arr[pos], arr[wall] = arr[wall], arr[pos]
            wall += 1
    arr[wall], arr[last] = arr[last], arr[wall]
    return wall


x = [7, 10, 9, 1, 6, 4, 3, 2, 5, 8]
print(quick_sort(x))


def new_tree_from_grid(grid, split_values_x, split_values_y, feature_indices):
    """
    Build XGBoost-compatible tree structure from a 2D grid of alpha values.
    """

    base_weights = []
    left_children = []
    right_children = []
    split_indices = []
    split_conditions = []
    parents = []
    default_left = []
    split_type = []
    loss_changes = []
    sum_hessian = []

    node_counter = [0]  # Mutable state to track node IDs

    def recurse(x_lo, x_hi, y_lo, y_hi, parent=-1, prefer_axis=0):
        i = node_counter[0]

        # Base case (leaf)
        if (x_hi - x_lo == 1) and (y_hi - y_lo == 1):
            alpha_value = grid[x_lo, y_lo]
            base_weights.append(float(alpha_value))
            left_children.append(-1)
            right_children.append(-1)
            split_indices.append(0)
            split_conditions.append(0.0)
            parents.append(parent)
            default_left.append(0)
            split_type.append(0)
            loss_changes.append(0.0)
            sum_hessian.append(1.0)
            node_counter[0] += 1
            return i

        # Choose axis to split (alternate or based on grid shape)
        if (prefer_axis == 0 and (x_hi - x_lo) > 1) or (y_hi - y_lo) == 1:
            axis = 0
            mid = (x_lo + x_hi) // 2
            split_val = split_values_x[mid - 1]
            left_id = recurse(x_lo, mid, y_lo, y_hi, parent=i, prefer_axis=1)
            right_id = recurse(mid, x_hi, y_lo, y_hi, parent=i, prefer_axis=1)
        else:
            axis = 1
            mid = (y_lo + y_hi) // 2
            split_val = split_values_y[mid - 1]
            left_id = recurse(x_lo, x_hi, y_lo, mid, parent=i, prefer_axis=0)
            right_id = recurse(x_lo, x_hi, mid, y_hi, parent=i, prefer_axis=0)

        # Internal node
        base_weights.append(0.0)
        left_children.append(left_id)
        right_children.append(right_id)
        split_indices.append(feature_indices[axis])
        split_conditions.append(float(split_val))
        parents.append(parent)
        default_left.append(1)  # assume default goes left
        split_type.append(0)
        loss_changes.append(0.0)
        sum_hessian.append(float((x_hi - x_lo) * (y_hi - y_lo)))  # optional

        node_counter[0] += 1
        return i

    recurse(0, grid.shape[0], 0, grid.shape[1])

    return {
        "base_weights": base_weights,
        "left_children": left_children,
        "right_children": right_children,
        "split_indices": split_indices,
        "split_conditions": split_conditions,
        "parents": parents,
        "default_left": default_left,
        "split_type": split_type,
        "loss_changes": loss_changes,
        "sum_hessian": sum_hessian,
        "categories": [],
        "categories_segments": [],
        "categories_sizes": [],
        "categories_nodes": [],
        "id": 0,
        "tree_param": {
            "num_deleted": "0",
            "num_feature": str(max(feature_indices) + 1),
            "num_nodes": str(len(base_weights)),
            "size_leaf_vector": "1",
        },
    }
