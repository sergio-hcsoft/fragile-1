from typing import Callable, Tuple

import numpy
import holoviews
from holoviews.streams import Buffer, Pipe
from scipy.interpolate import griddata


class Plot:

    name = ""

    def __init__(self, plot: Callable, data=None):
        self.plot = None
        self.init_plot(plot, data)

    def get_plot_data(self, data):
        raise NotImplementedError

    def init_plot(self, plot: Callable, data=None):
        data = self.get_plot_data(data)
        self.plot = plot(data)
        self.opts()

    def opts(self, *args, **kwargs):
        if self.plot is None:
            return
        self.plot = self.plot.opts(*args, **kwargs)


class StreamingPlot(Plot):
    name = ""

    def __init__(self, plot: Callable, stream=Pipe, data=None):
        self.data_stream = None
        self.epoch = 0
        self.init_stream(stream, data)
        super(StreamingPlot, self).__init__(plot=plot)

    def get_plot_data(self, data):
        return data

    def stream_data(self, data) -> None:
        data = self.get_plot_data(data)
        self.data_stream.send(data)
        self.epoch += 1

    def init_plot(self, plot: Callable, data=None) -> None:
        self.plot = holoviews.DynamicMap(plot, streams=[self.data_stream])
        self.opts()

    def init_stream(self, stream, data=None):
        self.epoch = 0
        data = self.get_plot_data(data)
        self.data_stream = stream(data=data)


class Curve(StreamingPlot):
    name = "curve"

    def __init__(self, buffer_length: int = 10000, index: bool = False, data=None,):
        def get_stream(data):
            return Buffer(data, length=buffer_length, index=index)

        super(Curve, self).__init__(stream=get_stream, plot=holoviews.Curve, data=data)

    def opts(
        self,
        title="",
        tools="default",
        xlabel: str = "x",
        ylabel: str = "y",
        shared_axes: bool = False,
        framewise: bool = True,
        axiswise: bool = True,
        normalize: bool = True,
        *args,
        **kwargs
    ):
        tools = tools if tools != "default" else ["hover"]
        self.plot = self.plot.opts(
            holoviews.opts.Curve(
                tools=tools,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                shared_axes=shared_axes,
                framewise=framewise,
                axiswise=axiswise,
                normalize=normalize,
                *args,
                *kwargs
            ),
            holoviews.opts.NdOverlay(
                normalize=normalize,
                framewise=framewise,
                axiswise=axiswise,
                shared_axes=shared_axes,
            ),
        )


class Histogram(StreamingPlot):
    name = "histogram"

    def __init__(self, n_bins: int = 20, data=None):
        self.n_bins = n_bins
        self.xlim = None
        super(Histogram, self).__init__(stream=Pipe, plot=self.plot_histogram, data=data)

    def plot_histogram(self, data):
        plot_data, xlim = data
        return holoviews.Histogram(plot_data).redim(x=holoviews.Dimension('x', range=xlim))

    def opts(
        self,
        title="",
        tools="default",
        xlabel: str = "x",
        ylabel: str = "count",
        shared_axes: bool = False,
        framewise: bool = True,
        axiswise: bool = True,
        normalize: bool = True,
        *args,
        **kwargs
    ):
        tools = tools if tools != "default" else ["hover"]
        self.plot = self.plot.opts(
            holoviews.opts.Histogram(
                tools=tools,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                shared_axes=shared_axes,
                framewise=framewise,
                axiswise=axiswise,
                normalize=normalize,
                *args,
                *kwargs
            ),
            holoviews.opts.NdOverlay(
                normalize=normalize,
                framewise=framewise,
                axiswise=axiswise,
                shared_axes=shared_axes,
            ),
        )

    def get_plot_data(self, data: numpy.ndarray):
        data[numpy.isnan(data)] = 0.
        return numpy.histogram(data, self.n_bins), self.xlim


class Bivariate(StreamingPlot):
    name = "bivariate"

    def __init__(self, data=None, *args, **kwargs):
        def bivariate(data):
            return holoviews.Bivariate(data, *args, **kwargs)

        super(Bivariate, self).__init__(stream=Pipe, plot=bivariate, data=data)

    def opts(
        self,
        title="",
        tools="default",
        xlabel: str = "x",
        ylabel: str = "y",
        shared_axes: bool = False,
        framewise: bool = True,
        axiswise: bool = True,
        normalize: bool = True,
        *args,
        **kwargs
    ):
        tools = tools if tools != "default" else ["hover"]
        self.plot = self.plot.opts(
            holoviews.opts.Bivariate(
                tools=tools,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                shared_axes=shared_axes,
                framewise=framewise,
                axiswise=axiswise,
                normalize=normalize,
                *args,
                **kwargs
            ),
            holoviews.opts.Scatter(
                fill_color="red",
                alpha=0.7,
                size=3.5,
                tools=tools,
                xlabel=xlabel,
                ylabel=ylabel,
                shared_axes=shared_axes,
                framewise=framewise,
                axiswise=axiswise,
                normalize=normalize,
                *args,
                **kwargs
            ),
            holoviews.opts.NdOverlay(
                normalize=normalize,
                framewise=framewise,
                axiswise=axiswise,
                shared_axes=shared_axes,
            ),
        )


class Landscape2D(StreamingPlot):
    name = "landscape"

    def __init__(self, n_points: int = 50, data=None):
        self.n_points = n_points
        self.xlim = (None, None)
        self.ylim = (None, None)
        super(Landscape2D, self).__init__(stream=Pipe, plot=self.plot_landscape, data=data)

    @staticmethod
    def plot_landscape(data):
        x, y, xx, yy, z, xlim, ylim = data
        zz = griddata((x, y), z, (xx, yy), method="linear")
        mesh = holoviews.QuadMesh((xx, yy, zz)).redim(x=holoviews.Dimension('x', range=xlim),
                                                      y=holoviews.Dimension('y', range=ylim),)
        contour = holoviews.operation.contours(mesh, levels=8)
        scatter = holoviews.Scatter((x, y))
        contour_mesh = mesh * contour * scatter
        return contour_mesh

    def opts(
        self,
        title="Distribution landscape",
        tools="default",
        xlabel: str = "x",
        ylabel: str = "y",
        shared_axes: bool = False,
        framewise: bool = True,
        axiswise: bool = True,
        normalize: bool = True,
        cmap: str = "viridis",
        height: int = 350,
        width: int = 350,
        *args,
        **kwargs
    ):
        tools = tools if tools != "default" else ["hover"]
        self.plot = self.plot.opts(
            holoviews.opts.QuadMesh(
                cmap=cmap,
                colorbar=True,
                title=title,
                bgcolor="lightgray",
                tools=tools,
                xlabel=xlabel,
                ylabel=ylabel,
                shared_axes=shared_axes,
                framewise=framewise,
                axiswise=axiswise,
                normalize=normalize,
                height=height,
                width=width,
                *args,
                **kwargs
            ),
            holoviews.opts.Contours(
                cmap=["black"],
                line_width=1,
                alpha=0.9,
                tools=tools,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                show_legend=False,
                shared_axes=shared_axes,
                framewise=framewise,
                axiswise=axiswise,
                normalize=normalize,
                *args,
                **kwargs
            ),
            holoviews.opts.Scatter(
                fill_color="red",
                alpha=0.7,
                size=3.5,
                tools=tools,
                xlabel=xlabel,
                ylabel=ylabel,
                shared_axes=shared_axes,
                framewise=framewise,
                axiswise=axiswise,
                normalize=normalize,
                *args,
                **kwargs
            ),
            holoviews.opts.NdOverlay(
                normalize=normalize,
                framewise=framewise,
                axiswise=axiswise,
                shared_axes=shared_axes,
            ),
        )

    def get_plot_data(self, data: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]):
        x, y, z = (data[:, 0], data[:, 1], data[:, 2]) if isinstance(data, numpy.ndarray) else data
        # target grid to interpolate to
        xi = numpy.linspace(x.min(), x.max(), self.n_points)
        yi = numpy.linspace(y.min(), y.max(), self.n_points)
        xx, yy = numpy.meshgrid(xi, yi)
        return x, y, xx, yy, z, self.xlim, self.ylim
