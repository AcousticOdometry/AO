import ao

import pandas as pd


def test_generate_wheel_odometry(odometry_ground_truth):
    wheel_odom = ao.evaluate.generate_wheel_odometry(
        odometry_ground_truth['Vw'], wheel_radius=0.1
        )
    assert 'Vw' in wheel_odom.columns
    assert 'Vx' in wheel_odom.columns
    assert 'tx' in wheel_odom.columns
    assert 'X' in wheel_odom.columns


def test_odometry(odometry_estimations, odometry_ground_truth):
    for odom in odometry_estimations:
        evaluation = ao.evaluate.odometry(odom, odometry_ground_truth)
        assert any(['ATE' in col for col in evaluation.columns])
        assert any(['RPE' in col for col in evaluation.columns])
        assert any(['APE' in col for col in evaluation.columns])
        assert pd.notna(evaluation).all().all()


def test_odometry_comparison(odometry_estimations, odometry_ground_truth):
    pass