import os
import pathlib
from abc import abstractmethod
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
from toga.widgets.canvas.context import Context
from toga.widgets.canvas.drawingobject import DrawingObject

clr.AddReference("System.Drawing")  # noqa
from System.Drawing import Bitmap as WinBmap  # noqa
from System.Drawing import Image as WinImage  # noqa
from System.Drawing import Size as WinSize  # noqa

PathLikeT: TypeAlias = str | os.PathLike
BytesLikeT: TypeAlias = bytes | bytearray | memoryview
ImageLikeT: TypeAlias = Any
ImageContentT: TypeAlias = PathLikeT | BytesLikeT | ImageLikeT


class CustomBackgroundBase(DrawingObject):
    """
    Custom DrawingObject base to set up the background of toga.Canvas.
    """

    @abstractmethod
    def __init__(self, image: toga.Image, id: str, style: StyleT, **kwargs: Any):
        pass

    @abstractmethod
    def _draw(self, impl, **kwargs) -> None:
        pass


class CustomContextBase(Context):
    """
    Custom Context to to be used with the custom canvas with variable background.
    """

    @abstractmethod
    def __init__(self, canvas: toga.Canvas, **kwargs):
        pass
        # super().__init__(canvas, kwargs=kwargs)

    @abstractmethod
    def background(
        self, image, id, style, background_class: type, **kwargs
    ) -> CustomBackgroundBase:
        pass


class CustomCanvas(toga.ImageView, toga.Canvas):
    """
    Sets up a custom canvas widget with variale background.
    """

    @abstractmethod
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
        pass

    @abstractmethod
    def _create(self) -> Any:
        pass

    @property
    @abstractmethod
    def image(self) -> toga.Image | None:
        pass

    @image.setter
    @abstractmethod
    def image(self, image: ImageContentT | pathlib.Path | str) -> None:
        pass

    @property
    @abstractmethod
    def on_press(self) -> OnTouchHandler | Callable:
        pass

    @on_press.setter
    @abstractmethod
    def on_press(self, handler: OnTouchHandler) -> None:
        pass

    @property
    @abstractmethod
    def on_release(self) -> OnTouchHandler | Callable:
        pass

    @on_release.setter
    @abstractmethod
    def on_release(self, handler: OnTouchHandler) -> None:
        pass

    @property
    @abstractmethod
    def on_drag(self) -> OnTouchHandler | Callable:
        pass

    @on_drag.setter
    @abstractmethod
    def on_drag(self, handler: OnTouchHandler) -> None:
        pass
