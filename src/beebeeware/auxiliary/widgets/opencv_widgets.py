import os
import pathlib
from typing import Generator  # noqa
from typing import OrderedDict  # noqa
from typing import Any, AsyncGenerator, Callable, TypeAlias, Union  # noqa

import clr
import toga
import toga.handlers
import toga.paths
import toga.sources
import toga.validators
from toga.widgets.base import StyleT
from toga.widgets.canvas import OnResizeHandler, OnTouchHandler

from .bases import CustomBackgroundBase, CustomCanvas, CustomContextBase

clr.AddReference("System.Drawing")  # noqa
from System.Drawing import Bitmap as WinBmap  # noqa
from System.Drawing import Image as WinImage  # noqa
from System.Drawing import Size as WinSize  # noqa

PathLikeT: TypeAlias = str | os.PathLike
BytesLikeT: TypeAlias = bytes | bytearray | memoryview
ImageLikeT: TypeAlias = Any
ImageContentT: TypeAlias = PathLikeT | BytesLikeT | ImageLikeT


class CVBackground(CustomBackgroundBase):
    def _draw(self, impl, **kwargs):
        impl.push_context(**kwargs)
        toga.ImageView(self.image, self.id, self.style, kwargs)
        impl.pop_context(**kwargs)


class CVContext(CustomContextBase):
    def background(
        self, image, id, style, background_class=CVBackground, **kwargs
    ) -> CVBackground:
        background = background_class(image, id, style, **kwargs)
        self.append(background)
        return background


class CVView(CustomCanvas):
    def __init__(
        self,
        image: ImageContentT | None = None,
        id: str | None = None,
        height: int | None = None,
        width: int | None = None,
        style: StyleT | None = None,
        on_resize: OnResizeHandler | Callable | None = None,
        on_press: OnTouchHandler | Callable | None = None,
        on_release: OnTouchHandler | Callable | None = None,
        on_drag: OnTouchHandler | Callable | None = None,
        image_path: str | pathlib.Path | None = None,
        scaling_factor: float | None = None,
        source_path: str | pathlib.Path | None = None,
        **kwargs,
    ):
        """Create a new image view.

        :param image: The image to display. Can be any valid
            [image content][toga.images.ImageContentT] type; or [`None`][] to
            display no image.
        :param id: The ID for the widget.
        :param style: A style object. If no style is provided, a default style will be
            applied to the widget.
        :param kwargs: Initial style properties.
        """
        super().__init__(
            image=image,
            id=id,
            height=height,
            width=width,
            style=style,
            on_resize=on_resize,
            on_press=on_press,
            on_release=on_release,
            on_drag=on_drag,
            image_path=image_path,
            scaling_factor=scaling_factor,
            source_path=source_path,
            **kwargs,
        )

    def _create(self) -> Any:
        return self.factory.Canvas(interface=self)
