"""Classes and functions related to human targeted visualization."""

from typing import Generic, Protocol

import cv2
import matplotlib
import numpy

from matplotlib import colors
from matplotlib import pyplot
from numpy.typing import NDArray

from reinfocus import histories
from reinfocus.graphics import render
from reinfocus.environments import episode_ender
from reinfocus.environments.types import ObservationT_contra, StateT_contra


def fading_colours(cmap: colors.Colormap, max_n: int, n: int, p: int = 2):
    """Makes an array of colours that gradually fade from cmap(1) to cmap(1 / max_n) as n
    goes from 1 to max_n.

    Args:
        cmap: The colour map that defines the colours involved in the fade.
        max_n: The largest number of colours that can be requested.
        n: How many colours are fading from cmap(1) to cmap(0). If n is max_n, the fade
            will reach from cmap(1) to cmap((1 / max_n) ** p), if n is smaller, the fade
            will be proportionally shorter.
        p: Each step of the fade is 1 / n ** p times smaller than the one before.

    Returns:
        An array of n colours fading from cmap(1) to cmap(1 / max_n) as n goes from 1 to
        max_n."""

    samples = numpy.linspace(1 - (n - 1) / max_n, 1, n) ** p
    colours = cmap(samples)
    colours[:, -1] = samples
    return colours


class IEpisodeVisualizer(Protocol, Generic[ObservationT_contra, StateT_contra]):
    # pylint: disable=too-few-public-methods, unnecessary-ellipsis
    """The base that episode visualizers must follow."""

    def step(
        self,
        states: StateT_contra,
        observations: NDArray[ObservationT_contra],
        indices: NDArray[numpy.bool_] | None = None,
    ):
        """Update the visualizer with a batch of states and observations. Should only be
        called once per timestep.

        Args:
            states: The new states that were reached on the current timestep.
            observations: The observations seen during states.
            indices: Which environments to update."""

        ...

    def reset(
        self,
        states: StateT_contra,
        observations: NDArray[ObservationT_contra],
        indices: NDArray[numpy.bool_] | None = None,
    ):
        """Informs the visualizer that some episodes have restarted.

        Args:
            states: The first states of the new episodes that reset marks the start of.
            observations: The first observations of the new states of the new episodes.
            indices: None, or a numpy array of one boolean per environment, where each
                element is True if that environment has just been reset. If None, all
                environments are considered reset."""

        ...

    def visualize(self) -> NDArray[numpy.uint8]:
        """Produces an image representing the ongoing episodes.

        Returns:
            Some image representing the ongoing episodes."""

        ...


class HistoryVisualizer(IEpisodeVisualizer):
    # pylint: disable=too-many-instance-attributes
    """Produces images of a batch of agents' performance in a number of focus
    environments. Renderings of the environments are on the left, while plots of the
    agents' performance over time is on the right. Each environments image is stacked into
    a vertical list."""

    def __init__(
        self,
        num_envs: int,
        target_index: int,
        focus_plane_index: int,
        focus_value_index: int,
        renderer: render.FastRenderer,
        limits: tuple[float, float],
        ender: episode_ender.IEpisodeEnder[NDArray[numpy.float32]] | None = None,
        history_length: int = 10,
        target_radius: float | None = None,
    ):
        # pylint: disable=too-many-arguments
        """Creates a HistoryVisualizer.

        Args:
            num_envs: The number of environments to visualize.
            target_index: The index of the target location in the state.
            focus_plane_index: The index of the focus plane location in the state.
            focus_value_index: The index of the focus value in the observation.
            ender: The episode ender whose status will be displayed.
            world_handle: The worlds to render.
            limits: The minimum and maximum position of the target and focus plane.
            history_length: The number of moves into the past which will be plotted.
            target_radius: Will render a column showing where moves will be within this
                distance from the target."""

        self._num_envs = num_envs
        self._target_index = target_index
        self._focus_plane_index = focus_plane_index
        self._focus_value_index = focus_value_index
        self._limits = limits
        self._history_length = history_length
        self._target_radius = target_radius
        self._ender = ender
        self._renderer = renderer

        self._current_moves = numpy.zeros(num_envs, dtype=numpy.int32)
        self._targets = numpy.zeros(num_envs, dtype=numpy.float32)
        self._move_histories = histories.Histories(num_envs, history_length)
        self._focus_histories = histories.Histories(num_envs, history_length)

    def step(
        self,
        states: NDArray[numpy.float32],
        observations: NDArray[numpy.float32],
        indices: NDArray[numpy.bool_] | None = None,
    ):
        """Update the visualizer with a batch of states and observations. Should only be
        called once per timestep.

        Args:
            states: The new states that were reached on the current timestep.
            observations: The observations seen during states.
            indices: Which environments to update."""

        if indices is None:
            indices = numpy.full(self._num_envs, True)

        self._current_moves[indices] += 1

        self._move_histories.append_events(states[:, self._focus_plane_index], indices)
        self._focus_histories.append_events(
            observations[:, self._focus_value_index], indices
        )

    def reset(
        self,
        states: NDArray[numpy.float32],
        observations: NDArray[numpy.float32],
        indices: NDArray[numpy.bool_] | None = None,
    ):
        """Informs the visualizer that some episodes have restarted.

        Args:
            states: The first states of the new episode that reset marks the start of.
            observations: The first observations of the new states of the new episodes.
            indices: None, or a numpy array of one boolean per environment, where each
                element is True if that environment has just been reset. If None, all
                environments are considered reset."""

        if indices is None:
            indices = numpy.full(self._num_envs, True)

        self._current_moves[indices] = 0

        self._targets[indices] = states[:, self._target_index]

        self._move_histories.reset(indices)
        self._move_histories.append_events(states[:, self._focus_plane_index], indices)

        self._focus_histories.reset(indices)
        self._focus_histories.append_events(observations[:, self._focus_value_index])

    def visualize(self) -> NDArray[numpy.uint8]:
        # pylint: disable=no-member
        """Produces a vertical stack of images representing the ongoing episode of each
        environment. Each image has a rendering of the scene on the left and a graph of
        performance on the right.

        Returns:
            A single image showing the performance in all environments."""

        renderings = self._renderer.render(600)

        graphs = [self._visualize_single_history(i) for i in range(self._num_envs)]

        return cv2.vconcat(
            [cv2.hconcat([r, g]) for r, g in zip(renderings, graphs)]
        ).astype(numpy.uint8)

    def _visualize_single_history(
        self,
        env_index: int,
        frame_height: int = 600,
    ) -> NDArray[numpy.uint8]:
        # pylint: disable=no-member,too-many-locals
        """Produces an image representing the ongoing episode of some environment. It has
        a rendering of the scene on the left and a graph of performance on the right.

        Args:
            env_index: The index of the environment that will show it's performance.
            frame_height: How many pixels tall the resulting image will be.

        Returns:
            A single image showing the performance in one environment."""

        focus_history = self._focus_histories.get_history(env_index)
        move_history = self._move_histories.get_history(env_index)
        target = self._targets[env_index]

        n_focus_history = len(focus_history)

        figure, axes = pyplot.subplots()
        axes.set_xlim(*self._limits)
        axes.set_ylim(-1.0, 1.0)

        x_label = f"focus position {self._current_moves[env_index]}\n"

        if self._ender is not None:
            x_label += self._ender.status(env_index)

        axes.set_xlabel(x_label)
        axes.set_ylabel("focus value")

        axes.axvline(x=target, linestyle=":", color="darkorange", label="target")

        if self._target_radius is not None and self._target_radius > 0.0:
            axes.axvspan(
                target - self._target_radius,
                target + self._target_radius,
                edgecolor="darkorange",
                facecolor=("darkorange", 0.1),
                linestyle=(0, (5, 10)),
            )

        fading_blues = fading_colours(
            matplotlib.colormaps["Blues"], self._history_length, n_focus_history
        )

        old_move_and_focus = None

        for i, move_and_focus in enumerate(zip(move_history, focus_history)):
            colour = fading_blues[i]

            pyplot.plot(
                *move_and_focus,
                color=colour,
                zorder=i,
                marker=".",
                label="focus" if i == n_focus_history - 1 else "",
            )

            if old_move_and_focus is not None:
                axes.annotate(
                    "",
                    xy=move_and_focus,
                    xycoords="data",
                    xytext=old_move_and_focus,
                    textcoords="data",
                    arrowprops={
                        "arrowstyle": "->",
                        "color": colour,
                        "shrinkA": 5,
                        "shrinkB": 5,
                        "connectionstyle": "arc3,rad=0.1",
                    },
                )

            old_move_and_focus = move_and_focus

        figure.legend(loc="lower right")

        figure.tight_layout()

        figure.canvas.draw()

        image_array = numpy.array(figure.canvas.buffer_rgba())[:, :, :3]  # type: ignore

        pyplot.close(figure)

        return cv2.resize(
            image_array,
            (
                int(frame_height * image_array.shape[1] / image_array.shape[0]),
                frame_height,
            ),
        ).astype(numpy.uint8)
