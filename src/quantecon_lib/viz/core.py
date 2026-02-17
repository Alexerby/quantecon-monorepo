import matplotlib.pyplot as plt
import os
import numpy as np

# Get path to your .mplstyle relative to the library root
STYLE_PATH = os.path.join(os.path.dirname(__file__), "../styles/sleak.mplstyles")

def use_quantecon_style():
    """Apply the library's custom matplotlib style."""
    if os.path.exists(STYLE_PATH):
        plt.style.use(STYLE_PATH)
    else:
        # Fallback to a clean built-in style if file is missing
        plt.style.use('seaborn-v0_8-muted')

def plot_spline_fit(X, y, model, label="Spline Fit"):
    """Standardized APA-style plot for basis models."""
    use_quantecon_style()
    
    # Generate smooth line for prediction
    X_smooth = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
    y_smooth = model.predict(X_smooth)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot Data
    ax.scatter(X, y, alpha=0.2, color='gray', label="Observations")
    
    # Plot Spline
    ax.plot(X_smooth, y_smooth, label=label)
    
    # Plot Knots
    if hasattr(model, 'knots') and model.knots is not None:
        knot_y = model.predict(model.knots.reshape(-1, 1))
        ax.scatter(model.knots, knot_y, color='black', zorder=5, label="Knots")
        
        # Add subtle vertical lines for knots to show regions
        for k in model.knots:
            ax.axvline(k, color='gray', linestyle='--', alpha=0.3, linewidth=1)
            
    ax.set_title(f"{model.__class__.__name__} Visualization")
    ax.legend(frameon=True)
    
    return fig, ax
