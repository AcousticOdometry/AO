import ao
import pandas as pd

from typing import Union, List, Tuple, Optional

from matplotlib import pyplot as plt
from matplotlib import ticker


def _process_plots_input(plots: List[Tuple[pd.DataFrame, Union[dict, str]]]):
    if not plots:
        raise ValueError(f'Nothing to plot: plots={plots}')
    elif not isinstance(plots, list):
        raise TypeError(
            f"`plots` must be a list or a pd.DataFrame, not {type(plots)}"
            )
    elif not isinstance(plots[0], tuple):
        raise TypeError(
            f"`plots` must be a list of tuples, not a list of {type(plots[0])}"
            )
    elif 1 > len(plots[0]) > 2:
        raise ValueError('Tuples inside `plots` must be of length 2')
    for i, (odom, kwargs) in enumerate(plots):
        if isinstance(kwargs, str):
            plots[i] = (odom, {'label': kwargs})
        elif not isinstance(kwargs, dict):
            raise TypeError(
                f"`kwargs` must be a string or a dict, not {type(kwargs)}"
                )
    return plots


def format_time_xaxis(
        ax: plt.Axes, start_timestamp: float, end_timestamp: float
    ) -> None:
    ax.set_xlim(start_timestamp, end_timestamp)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{int(x - start_timestamp)}")
        )
    ax.set_xlabel('Time [s]')


def odometry(
    odom: pd.DataFrame,
    *,
    ax: Optional[plt.Axes] = None,
    suptitle: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
    if not isinstance(odom, pd.DataFrame):
        raise TypeError(f"`odom` must be a pandas.DataFrame not {type(odom)}")
    ax = odom.plot(ax=ax, y=['Vx'], ylabel='Speed [m/s]', color='orange')
    odom.plot(ax=ax, y=['X'], secondary_y=True).set_ylabel('Position [m]')
    format_time_xaxis(ax, odom.index.min(), odom.index.max())
    fig = ax.get_figure()
    # Add supertitle
    if suptitle:
        fig.suptitle(suptitle)
    return fig, ax


def odometry_comparison(
    plots: List[Tuple[pd.DataFrame, Union[dict, str]]],
    *,
    ax_speed: Optional[plt.Axes] = None,
    ax_position: Optional[plt.Axes] = None,
    suptitle: Optional[str] = None,
    show_legend: bool = True,
    **subplots_kwargs,
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
    # Check plots input
    plots = _process_plots_input(plots)
    # Check X axis limits
    start_timestamp = min([odom.index.min() for odom, _ in plots])
    end_timestamp = max([odom.index.max() for odom, _ in plots])
    # Generate new axes or use existing
    if not ax_speed and not ax_position:
        fig, (ax_speed, ax_position) = plt.subplots(1, 2, **subplots_kwargs)
    elif ax_speed:
        fig = ax_speed.get_figure()
    else:
        fig = ax_position.get_figure()
    # Plot
    for odom, plot_kwargs in plots:
        if ax_position:
            odom.plot(ax=ax_position, y='X', **plot_kwargs, legend=False)
        if ax_speed:
            ax_speed.plot(
                odom.index,
                odom.Vx,
                'x--',
                markersize=3,
                alpha=0.5,
                **plot_kwargs
                )
    # Format X axis
    if ax_speed:
        ax_speed.set_ylabel('Speed [m/s]')
        format_time_xaxis(ax_speed, start_timestamp, end_timestamp)
    if ax_position:
        ax_position.set_ylabel('Position [m]')
        format_time_xaxis(ax_position, start_timestamp, end_timestamp)
    # Add supertitle
    if suptitle:
        fig.suptitle(suptitle)
    # Add legend
    if show_legend:
        fig.legend(
            *(ax_speed
              if ax_speed else ax_position).get_legend_handles_labels(),
            bbox_to_anchor=(0.5, 0),
            loc='upper center'
            )
    return fig, (ax_speed, ax_position)


def evaluation(
    data: pd.DataFrame,
    ground_truth: Optional[pd.DataFrame] = None,
    evaluate_kwargs: dict = {},
    *,
    fig: Optional[plt.Figure] = None,
    axs: Optional[List[plt.Axes]] = None,
    show_only: Optional[List[str]] = ['RPE', 'APE'],
    show_mean_on: List[str] = ['ATE', 'RPE'],
    **plot_kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"`data` must be a pandas.DataFrame not {type(data)}")
    if ground_truth is None:
        # Assume that the provided data is directly the evaluation
        evaluation = data
    else:
        evaluation = ao.evaluate.odometry(
            data, ground_truth, **evaluate_kwargs
            )
    if show_only:
        evaluation_columns = [
            col for col in evaluation.columns if col in show_only
            ]
    else:
        evaluation_columns = evaluation.columns
    if axs is None:
        if fig is None:
            fig, axs = plt.subplots(1, len(evaluation_columns))
        else:
            axs = fig.subplots(1, len(evaluation_columns))
    else:
        fig = axs[0].get_figure()
    if len(axs) != len(evaluation_columns):
        raise ValueError(
            "`axs` must be a list of axes of the same length as the provided "
            f"`evaluation` columns: {len(axs)} != len({evaluation_columns})"
            )
    for ax, col in zip(axs, evaluation_columns):
        ax.set_title(col)
        line = ax.plot(
            evaluation.index.to_numpy(), evaluation[col].to_numpy(),
            **plot_kwargs
            )[0]
        if col in show_mean_on:
            ax.axhline(
                evaluation[col].mean(),
                color=line.get_color(),
                **{
                    **{k: v
                       for k, v in plot_kwargs.items() if k != 'color'},
                    'ls': '--',
                    'label': None,
                    }
                )
        ax.set_ylabel(f"Error [{evaluation.attrs['unit'][col]}]")
    return fig, axs


def evaluation_comparison(
    plots: List[Tuple[pd.DataFrame, Union[dict, str]]],
    ground_truth: Optional[pd.DataFrame] = None,
    evaluate_kwargs: dict = {},
    *,
    fig: Optional[plt.Figure] = None,
    axs: Optional[List[plt.Axes]] = None,
    suptitle: Optional[str] = None,
    show_legend: bool = True,
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
    # Check plots input
    plots = _process_plots_input(plots)
    # Check X axis limits
    start_timestamp = min([odom.index.min() for odom, _ in plots])
    end_timestamp = max([odom.index.max() for odom, _ in plots])
    # Plot
    for i, (data, plot_kwargs) in enumerate(plots):
        fig, axs = evaluation(
            data,
            ground_truth,
            evaluate_kwargs=evaluate_kwargs,
            fig=fig,
            axs=axs,
            color=f"C{(i + 1) % 10}",  # Skip the first color
            **plot_kwargs
            )
    # Format X axis
    for ax in axs:
        format_time_xaxis(ax, start_timestamp, end_timestamp)
    # Add supertitle
    if suptitle:
        fig.suptitle(suptitle)
    # Add legend
    if show_legend:
        fig.legend(
            *axs[0].get_legend_handles_labels(),
            bbox_to_anchor=(0.5, 0),
            loc='upper center'
            )
    return fig, axs