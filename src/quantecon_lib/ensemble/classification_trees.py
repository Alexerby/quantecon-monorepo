import numpy as np
from ucimlrepo import fetch_ucirepo 


def find_best_split(X, y): 
    """Greedy search strategy."""
    best_rss = float("inf")
    split_info = None
    

    # For every feature in { x_1, x_2,..., x_p }
    for feature_idx in range(X.shape[1]):

        # 1. Iterative scanning
        unique_vals = np.unique(X[:, feature_idx]) 
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2 
       
       # For every split 
        for s in thresholds:

            # 2. Split the data into two distinct regions
            left_mask = X[:, feature_idx] < s
            right_mask = ~left_mask
            
            y_l, y_r = y[left_mask], y[right_mask]
            
            # Step 3: Determine if RSS in this iteration is the best  
            # one so far. If that's the case use this split.
            m_l, m_r = np.mean(y_l), np.mean(y_r)
            rss = np.sum((y_l - m_l)**2) + np.sum((y_r - m_r)**2)
            
            if rss < best_rss:
                best_rss = rss
                split_info = {
                    'feature_idx': feature_idx,
                    'threshold': s,
                    'left_mask': left_mask,
                    'right_mask': right_mask,
                    'rss': rss,
                    'means': (m_l, m_r)
                }

    return split_info

def build_tree(X, y, current_depth, max_depth, features):

    # 1. STOP CONDITIONS (Base Cases)
    # We stop splitting if we reach the depth limit (height guard) 
    # or if all remaining target values are identical (perfect purity).
    if current_depth >= max_depth or len(np.unique(y)) <= 1:
        return {"is_leaf": True, "prediction": np.mean(y)}

    # 2. THE SEARCH (Divide the Predictor Space)
    # Call our helper function to find the single best feature and threshold.
    split = find_best_split(X.values, y.values)
    
    # If the features are identical and no valid split is possible, 
    # we stop and turn this node into a terminal leaf.
    if not split:
        return {"is_leaf": True, "prediction": np.mean(y)}

    # 3. RECURSION (The Hand-off)
    # We identify the feature name and then call build_tree again for both 
    # the left and right subsets, increasing the depth counter by 1.
    feature = features[split['feature_idx']] # simply for identifying the feature name
    
    return { 
        "is_leaf": False, # This is a Decision Node, not a leaf
        "feature": feature,
        "threshold": split['threshold'],
        
        # The recursion dives down the left branch entirely before 
        # starting the right branch.
        "left": build_tree(X[split['left_mask']], y[split['left_mask']], 
                           current_depth + 1, max_depth, features),
                           
        "right": build_tree(X[split['right_mask']], y[split['right_mask']], 
                            current_depth + 1, max_depth, features)
    }

def visualize_tree(node, prefix="", is_left=True, is_root=True):
    """Visualizes the tree with icons to distinguish splits from terminal leaves."""
    
    # Base Case: Terminal Node (Leaf)
    if node.get("is_leaf"):
        connector = "└── " if not is_root else ""
        # 🍃 represents the final region R_j where the mean prediction is made
        print(f"{prefix}{connector} (LEAF) Region Prediction: {node['prediction']:.2f}")
        return

    # Recursive Step: Internal Node (Split)
    # 🌲 represents the decision point where space is partitioned
    connector = "├── " if not is_root else ""
    print(f"{prefix}{connector}Split: {node['feature']} < {node['threshold']:.2f}")

    # Adjust prefix for child branches
    new_prefix = prefix + ("│   " if not is_root else "")
    
    # Recurse through the binary branches
    visualize_tree(node["left"], new_prefix, is_left=True, is_root=False)
    visualize_tree(node["right"], new_prefix, is_left=False, is_root=False)

