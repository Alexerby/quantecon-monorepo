import numpy as np

def bootstrap(data, estimator, B=5000):
    """
    Performs non-parametric bootstrapping.
    
    Parameters:
    data (array): The original sample
    estimator (function): The statistic to calculate (e.g., np.mean)
    B (int): Number of bootstrap iterations
    """
    n = len(data)
    boot_estimates = []

    for i in range(B):
        # 1. Resample with replacement
        resample = np.random.choice(data, size=n, replace=True)
        
        # 2. Calculate the statistic for this resample
        stat = estimator(resample)
        
        # 3. Store the result
        boot_estimates.append(stat)
    
    # Standard Error is the Std Dev of the bootstrap distribution
    boot_se = np.std(boot_estimates, ddof=1)
    boot_mean = np.mean(boot_estimates)
    
    return boot_mean, boot_se, np.array(boot_estimates)
