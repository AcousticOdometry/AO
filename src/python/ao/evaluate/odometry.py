import numpy as np
import pandas as pd


def generate_wheel_odometry(
    measurements: pd.Series,
    wheel_radius: float,
    *,
    degrees: bool = True,
    ) -> pd.DataFrame:
    """Generates basic wheel odometry from wheel angular speed measurements and
        the wheel radius.

    Args:
        measurements (pd.Series): Wheel angular speed measurements with the
            corresponding timestamp as index. The value of the measurements is
            considered to be degrees per second if `degrees` is True,
            otherwise, radians per second.
        wheel_radius (float): Radius of the wheel in meters.
        degrees (bool, optional): Whether the measurements unit is in degrees.
            Defaults to True. 

    Returns:
        pd.DataFrame: Odometry data with columns 'tx', 'X' and timestamps as
            index.
    """
    wheel_odom = pd.DataFrame(index=measurements.index)
    wheel_odom['Vw'] = measurements.values
    if degrees:
        wheel_odom['Vw'] = np.deg2rad(measurements.values)
    else:
        wheel_odom['Vw'] = measurements.values
    wheel_odom['Vx'] = wheel_radius * wheel_odom['Vw']
    wheel_odom['tx'] = wheel_odom.index.to_series().diff() * wheel_odom['Vx']
    wheel_odom['X'] = wheel_odom['tx'].cumsum()
    return wheel_odom


def odometry(
    odom: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    delta_seconds: float = 1,
    ) -> pd.DataFrame:
    """Evaluates the provided odometry data against the ground truth

    Args:
        odom (pd.DataFrame): Odometry data with at least columns 'tx', 'X' and
            timestamps as index.
        ground_truth (pd.DataFrame): Ground truth data with at least columns
            'tx', 'X' and timestamps as index.
        delta_seconds (float, optional): Width of the time window for the
            Relative Position Error in seconds. Defaults to 1.

    Returns:
        pd.DataFrame: Evaluated metrics with columns 'ATE', 'APE' and 'RPE'
            corresponding to 'Absolute Translation Error', 'Absolute Position
            Error', and 'Relative Position Error' respectively.
    """
    ts = odom.index.to_numpy()
    gt_ts = ground_truth.index.to_numpy()
    # Interpolate ground truth to estimation timestamps
    sync_gt_tx = np.interp(ts, gt_ts, ground_truth['tx'].to_numpy())
    sync_gt_X = np.interp(ts, gt_ts, ground_truth['X'].to_numpy())
    evaluation = pd.DataFrame(index=ts)
    # ! Hacky way to provide additional info in column names
    evaluation.attrs['description'] = {}
    evaluation.attrs['unit'] = {}
    # Absolute Trajectory Error
    evaluation['ATE'] = np.absolute(sync_gt_tx - odom['tx'].to_numpy())
    evaluation.attrs['description']['ATE'] = (
        "Absolute Trajectory Error \n"
        f"{odom.index.to_series().diff().mean():.3f}s between estimations"
        )
    evaluation.attrs['unit']['ATE'] = 'm'
    # Relative Pose Error
    rel_X = odom['X'] - np.interp(ts - delta_seconds, ts, odom['X'])
    sync_gt_rel_X = sync_gt_X - np.interp(ts - delta_seconds, ts, sync_gt_X)
    evaluation['RPE'] = np.absolute(sync_gt_rel_X - rel_X.to_numpy())
    evaluation.attrs['description']['RPE'] = (
        f"Relative Pose Error\n{delta_seconds}s windows"
        )
    evaluation.attrs['unit']['RPE'] = 'm'
    # Absolute Pose Error
    evaluation['APE'] = np.absolute(sync_gt_X - odom['X'].to_numpy())
    evaluation.attrs['description']['APE'] = 'Absolute Pose Error'
    evaluation.attrs['unit']['APE'] = 'm'
    # Absolute Percentage Pose Error
    evaluation['APPE'] = evaluation['APE'] / sync_gt_X * 100
    evaluation.attrs['description']['APPE'] = (
        'Absolute Percentage\nPose Error'
        )
    evaluation.attrs['unit']['APPE'] = '%'
    return evaluation