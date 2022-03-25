import ao
import math
import pytest
import numpy as np

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
        temporal_integration=8 / 1000, # [s]
        ax=ax,
        pcolormesh_kwargs={'cmap': 'jet', 'vmin': -0.5},
        )
    ax.set_title('Ratemap')
    xlow, xhigh = ax.get_xlim()
    ax.set_xlim((xlow, xhigh - 0))
    f.colorbar(plot, cax=cax, orientation="horizontal")
    savefig(f)
