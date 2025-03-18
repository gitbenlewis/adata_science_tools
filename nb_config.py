import os
import sys
# import pandas try
try:
    import pandas as pd
    # pandas display options
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.options.display.max_seq_items = 2000
except ImportError as e:
    print(f"pandas import Error: {e}")

# import numpy, seaborn and matplotlib
try:
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    #%matplotlib inline
    # Custom colormap where lowest values are represented by grey good for umaps
    import matplotlib.colors as mcolors
    cmap = mcolors.ListedColormap(['gray'] + list(plt.cm.viridis(np.linspace(0, 1, 256))))
except ImportError as e:
    print(f"numpy, seaborn or matplotlib import Error: {e}")

# import scanpy
try:
    import scanpy as sc
    import anndata as ad
    # scanpy options
    sc.settings.verbosity = 1             # verbosity: errors (0), warnings (1), info (2), hints (3)
    sc.logging.print_header()
    sc.settings.set_figure_params(dpi=80, facecolor='white')
    #WRITE_CACHE=False # use false if  in github repo or cache files not in git ignore
except ImportError as e:
    print(f"scanpy import Error: {e}")   



