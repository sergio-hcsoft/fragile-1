import holoviews as hv
import numpy as np

from fragile.core.utils import resize_frame

ROOM_COORDS = {
    0: (3, 3),
    1: (4, 3),
    2: (5, 3),
    3: (2, 2),
    4: (3, 2),
    5: (4, 2),
    6: (5, 2),
    7: (6, 2),
    8: (1, 1),
    9: (2, 1),
    10: (3, 1),
    11: (4, 1),
    12: (5, 1),
    13: (6, 1),
    14: (7, 1),
    15: (0, 0),
    16: (1, 0),
    17: (2, 0),
    18: (3, 0),
    19: (4, 0),
    20: (5, 0),
    21: (6, 0),
    22: (7, 0),
    23: (8, 0),
}


def _save_images(data):
    best_plot, room_plot, n_iter = data
    hv.save(room_plot, filename="rooms_monte/image%05d.png" % n_iter)
    hv.save(best_plot, filename="monte_best/image%05d.png" % n_iter)


def plot_grid_over_obs(observation, grid) -> hv.NdOverlay:
    background = observation[50:, :].mean(axis=2).astype(bool).astype(int) * 255
    peste = resize_frame(grid.T[::1, ::1], 160, 160, "L")
    peste = peste / peste.max() * 255
    return hv.RGB(background) * hv.Image(peste).opts(alpha=0.7)


def create_memory_plots(displayed_rooms, memory):
    grid_plots = {}
    for k in displayed_rooms.keys():
        grid = memory[:, :, k]
        grid = resize_frame(grid.T, 160, 160, "L")
        grid = grid / grid.max() * 255
        room_grid_plot = hv.Image(grid).opts(
            xlim=(-0.5, 0.5),
            ylim=(-0.5, 0.5),
            xaxis=None,
            yaxis=None,
            title="Room %s" % k,
            cmap="fire",
            alpha=0.7,
        )
        grid_plots[k] = room_grid_plot
    return grid_plots


def plot_memories(displayed_rooms, memory):
    grid_plots = create_memory_plots(displayed_rooms, memory)
    memories = {ix: room * grid_plots[ix] for ix, room in displayed_rooms.items()}
    gridspace = hv.GridSpace(label="Explored rooms").opts(
        xaxis=None, yaxis=None, normalize=True, shared_xaxis=False, shared_yaxis=False
    )
    rows, cols = 4, 9
    grid_indexes = [(j, i) for i in reversed(range(rows)) for j in range(cols)]
    for ix in grid_indexes:
        gridspace[ix] = hv.Overlay(
            [hv.Image(np.ones((40, 40)) * 128).opts(cmap=["white"], shared_axes=False)]
        )
    for i, mem in memories.items():
        a, b = ROOM_COORDS[i]
        gridspace[a, b] = mem
    return gridspace


def save_memories(data):
    hv.extension("bokeh")
    *params, n_iter = data
    plot = plot_memories(*params)
    hv.save(plot, filename="monte_test/image%05d.png" % n_iter)
    del plot
