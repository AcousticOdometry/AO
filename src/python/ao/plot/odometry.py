import ao

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from typing import List, Tuple, Dict, Optional


def get_subplots(
    fig: Optional[plt.Figure] = None,
    axs: Optional[List[plt.Axes]] = None,
    **subplots_kwargs,
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
    raise NotImplementedError()


# TODO
def odometry(
    odom: pd.DataFrame,
    ground_truth: Optional[pd.DataFrame] = None,
    filter = lambda title: True,
    *,
    fig: Optional[plt.Figure] = None,
    axs: Optional[List[plt.Axes]] = None,
    suptitle: Optional[str] = None,
    **subplots_kwargs,
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
    raise NotImplementedError()


# TODO
def odometry_comparison(
    odoms: List[pd.DataFrame],
    filter = lambda title: True,
    *,
    fig: Optional[plt.Figure] = None,
    axs: Optional[List[plt.Axes]] = None,
    suptitle: Optional[str] = None,
    **subplots_kwargs,
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
    raise NotImplementedError()


# TODO
def odometry_evaluation(evaluation: pd.DataFrame, ):
    raise NotImplementedError()