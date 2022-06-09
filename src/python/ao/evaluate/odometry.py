import pandas as pd

# TODO
def odometry(
    odom: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    delta_seconds: float = 1,
    ) -> pd.Series:
    raise NotImplementedError()

def odometry_comparison(
    odoms: Dict[str, pd.DataFrame],
    ground_truth: pd.DataFrame,
    *,
    delta_seconds: float = 1,
    ) -> pd.DataFrame:
    raise NotImplementedError()