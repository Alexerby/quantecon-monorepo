from .classification_trees import build_tree

# 1. Calculate the residuals:
#   r_i1 = y_i - F_0(x_i)
#
# (THESE ARE TECHNICALLY THE NEGATIVE GRADIENTS OF OUR LOSS FUNCTION)

# 2. Fit a weak learner h_1(x) on those residuals. h_1(x) ~ r_i1

# 3. Update the model: F_1(x) = F_0(x) + \nu h_1(x)
# 4. REPEAT

def boosting(X, y, residuals):
    build_tree(X, y, current_depth=0, max_depth=6, features=X.columns)
    pass
