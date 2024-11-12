import pandas as pd
from ucimlrepo import fetch_ucirepo
import os

import numpy as np
from pandas.core.interchange.dataframe_protocol import DataFrame

from data import DataHandler
from hierarchical_clustering import HierarchicalClustering
from evaluation import calculate_and_save_silhouette, purity_score
from plot import save_dendrogram, save_silhouette_plot
if __name__=='__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, 'Assets', 'Dataset')
    output_dir = os.path.join(current_dir, 'Assets', 'Results')
    os.makedirs(output_dir, exist_ok=True)
    dataset_name = 'hepatitis'

