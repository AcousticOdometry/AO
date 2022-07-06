import ao
import math
import pytest

from matplotlib import pyplot as plt


@pytest.fixture()
def savefig(request, output_folder):

    def _savefig(fig: plt.Figure):
        fig.savefig(
            output_folder / (request.node.name.replace('test_', '') + '.png'),
            bbox_extra_artists=(*fig.legends, ),
            bbox_inches='tight',
            )

    return _savefig


def test_signal(audio_data, savefig):
    data, sample_rate = audio_data
    ax = ao.plot.signal(data, sample_rate)
    ax.set_title('Waveform')
    savefig(ax.figure)


def test_gammatonegram(audio_data, savefig):
    data, sample_rate = audio_data
    f, (ax, cax) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 0.3]})
    # Gammatonegram
    frame_length = 10  # [ms]
    frame_samples = math.ceil(frame_length / 1000 * sample_rate)
    plot, _ = ao.plot.gammatonegram(
        data,
        sample_rate,
        frame_samples,
        num_features=64,
        low_Hz=50,
        high_Hz=8000,
        temporal_integration=8 / 1000,  # [s]
        ax=ax,
        pcolormesh_kwargs={'cmap': 'jet'},
        )
    ax.set_title('Ratemap')
    xlow, xhigh = ax.get_xlim()
    ax.set_xlim((xlow, xhigh))
    f.colorbar(plot, cax=cax, orientation="horizontal")
    savefig(f)


def test_odometry(odometry_ground_truth, savefig):
    fig, _ = ao.plot.odometry(odometry_ground_truth, suptitle='Ground truth')
    savefig(fig)


def test_odometry_comparison(
        odometry_estimations, odometry_ground_truth, savefig
    ):
    plots = [(odometry_ground_truth, 'Ground truth')]
    plots.extend([(odom, {
        'label': f"microphone{i}"
        }) for i, odom in enumerate(odometry_estimations)])
    fig, _ = ao.plot.odometry_comparison(plots, suptitle='Multiple')
    savefig(fig)


def test_odometry_with_wheel_odometry(odometry_ground_truth, savefig):
    plots = [(odometry_ground_truth, 'Ground truth'),
             (
                 ao.evaluate.generate_wheel_odometry(
                     odometry_ground_truth['Vw'], wheel_radius=0.1
                     ), 'Wheel odometry'
                 )]
    fig, _ = ao.plot.odometry_comparison(plots, suptitle='Multiple')
    savefig(fig)


def test_odometry_type_error(odometry_estimations, odometry_ground_truth):
    with pytest.raises(TypeError):
        ao.plot.odometry_comparison(odometry_estimations)
    with pytest.raises(TypeError):
        plots = {
            'ground_truth': odometry_ground_truth,
            **{
                str(i): odom
                for i, odom in enumerate(odometry_estimations) if i != 4
                },
            }
        ao.plot.odometry_comparison(plots, suptitle='Dict of Odometry')


def test_evaluation(odometry_estimations, odometry_ground_truth, savefig):
    evaluation = ao.evaluate.odometry(
        odometry_estimations[0], odometry_ground_truth
        )
    fig, _ = ao.plot.evaluation(evaluation)
    savefig(fig)


def test_evaluation_shortcut(
        odometry_estimations, odometry_ground_truth, savefig
    ):
    fig, _ = ao.plot.evaluation(odometry_estimations[0], odometry_ground_truth)
    savefig(fig)


def test_evaluation_comparison(
        odometry_estimations, odometry_ground_truth, savefig
    ):
    fig, _ = ao.plot.evaluation_comparison(
        [(odom, str(i)) for i, odom in enumerate(odometry_estimations)],
        odometry_ground_truth,
        )
    savefig(fig)