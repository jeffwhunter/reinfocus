"""Classes and functions related to human targeted visualization."""

import cv2
import matplotlib
import numpy

from matplotlib import colors
from matplotlib import pyplot
from numpy.typing import NDArray

from reinfocus.graphics import render
from reinfocus.graphics import world


def fading_colours(cmap: colors.Colormap, max_n: int, n: int, p: int = 2):
    """Makes an array of colours that gradually fade from cmap(1) to cmap(1 / max_n) as n
    goes from 1 to max_n.

    Args:
        cmap: The colour map that defines the colours involved in the fade.
        max_n: The largest number of colours that can be requested.
        n: How many colours are fading from cmap(1) to cmap(0). If n is max_n, the fade
            will reach from cmap(1) to cmap(1 / max_n), if n is smaller, the fade will be
            proportionally shorter.
        p: Each step of the fade is 1 / n ** p times smaller than the one before.

    Returns:
        An array of n colours fading from cmap(1) to cmap(1 / max_n) as n goes from 1 to
        max_n."""

    samples = numpy.linspace(1 - (n - 1) / max_n, 1, n) ** p
    colours = cmap(samples)
    colours[:, -1] = samples
    return colours


class FocusHistoryVisualizer:
    """Produces a rendering on an agent's performance in a focus environment. Images of
    the environment are seen on the left, while a plot of the agent's performance over
    time is on the right."""

    def __init__(
        self,
        limits: tuple[float, float] = (5.0, 10.0),
        max_moves: int = 10,
        target_radius: float | None = None,
    ):
        """Creates a FocusHistoryVisualizer.

        Args:
            limits: The range over which the target and lens in some environment can vary.
            max_moves: The maximum number of moves to render in the performance history
                before truncating it.
            target_radius: The distance from the target to the edge of it's zone. If None
                or below zero, no zone is rendered.

        Returns:
            A FocusHistoryVisualizer."""

        self._limits = limits
        self._max_moves = max_moves
        self._target_radius = target_radius

        self._current_move = 0
        self._focus_history: list[tuple[float, ...]] = []

    def reset(self):
        """Resets the current visualization to the start of an episode."""

        self._current_move = 0
        self._focus_history = []

    def add_step(self, len_and_focus: tuple[float, float]):
        """Update the visualization with an agent's move and it's associated reward.

        Args:
            lens_and_focus: The len position and focus value in some step of a focus
                environment."""

        self._current_move += 1
        self._focus_history.append(len_and_focus)
        self._focus_history = self._focus_history[-self._max_moves :]

    def visualize(
        self,
        world_data: world.World,
        target: float,
        ender_status: str,
    ) -> NDArray:
        # pylint: disable=c-extension-no-member
        """Renders an easy (as possible so far) to understand image of an agent's
        performance in some focus environment. A rendering of the world will be on the
        left, while a plot of the agent's performance will be on the right.

        Args:
            world_data: The world to render.
            target: At what position the target is in the world.
            ender_status: How close the agent is to ending the episode early.

        Returns:
            An image of an agent's performance over time in some focus environment."""

        return cv2.hconcat(
            [
                render.render(
                    frame_shape=(600, 600),
                    world_data=world_data,
                    focus_distance=self._focus_history[-1][0],
                ),
                self._visualize_move_history(target, ender_status),
            ]
        )

    def _visualize_move_history(
        self,
        target: float,
        ender_status: str = "",
        frame_height: int = 600,
    ) -> NDArray:
        # pylint: disable=c-extension-no-member
        """Plots the performance of an agent in some focus environment.

        Args:
            target: At what position the target is in the world.
            ender_status: How close the agent is to ending the episode early.
            frame_height: How tall should the resulting visualization be in pixels.

        Returns:
            An image containing a plot of an agent's performance over time in some focus.
                environment."""

        n_focus_history = len(self._focus_history)

        figure, axes = pyplot.subplots()
        axes.set_xlim(*self._limits)

        x_label = f"lens position {self._current_move}"

        if ender_status != "":
            x_label += f", {ender_status}"

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
            matplotlib.colormaps["Blues"], self._max_moves, n_focus_history
        )

        for i in range(n_focus_history):
            move, focus = self._focus_history[i]
            colour = fading_blues[i]

            pyplot.plot(move, focus, color=colour, zorder=i, marker=".")

            if i > 0:
                axes.annotate(
                    "",
                    xy=self._focus_history[i],
                    xycoords="data",
                    xytext=self._focus_history[i - 1],
                    textcoords="data",
                    arrowprops={
                        "arrowstyle": "->",
                        "color": colour,
                        "shrinkA": 5,
                        "shrinkB": 5,
                        "connectionstyle": "arc3,rad=0.1",
                    },
                )

        figure.legend()

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
        )
