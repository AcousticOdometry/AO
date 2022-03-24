import ao
import math
import pytest

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

audio_urls = [
    r"https://staffwww.dcs.shef.ac.uk/people/N.Ma/resources/ratemap/t29_lwwj2n_m17_lgwe7s.wav",
    ]


@pytest.fixture(params=range(len(audio_urls)))
def audio_data(request):
    return ao.io.wave_read(audio_urls[request.param])


@pytest.fixture()
def savefig(request, output_folder):

    def _savefig(fig: plt.Figure):
        fig.tight_layout()
        fig.savefig(
            output_folder / (request.node.name.lstrip('test_') + '.png')
            )

    return _savefig


def test_gammatonegram(audio_data, savefig):
    data, sample_rate = audio_data
    fig, axs = plt.subplots(3, 1, gridspec_kw={'height_ratios': [1, 1, 0.05]})
    # Signal
    ao.plot.signal(data, sample_rate, ax=axs[0])
    axs[0].set_title('Waveform')
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
        ax=axs[1]
        )
    axs[1].set_title('Ratemap')
    fig.colorbar(plot, cax=axs[2], orientation="horizontal")
    savefig(fig)
