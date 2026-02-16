import matplotlib.pyplot as plt
import os
import sys

class SleakPlotter:
    COLORS = {
        'primary': '#0d6efd',
        'secondary': '#6c757d',
        'danger': '#dc3545',
        'bg': '#fafafa'
    }

    @staticmethod
    def apply_style():
        # Loads your local style file
        style_path = os.path.join(os.path.dirname(__file__), 'sleak.mplstyle')
        if os.path.exists(style_path):
            plt.style.use(style_path)
        else:
            plt.style.use('ggplot')

    @staticmethod
    def save_lab_plot(filename):
        # Determine the directory of the script calling this function
        caller_dir = os.path.dirname(sys.argv[0])
        output_dir = os.path.join(caller_dir, 'assets')
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        target_path = os.path.join(output_dir, filename)
        
        plt.tight_layout()
        plt.savefig(target_path, dpi=300, bbox_inches='tight')
        print(f"Saved to: {target_path}")
