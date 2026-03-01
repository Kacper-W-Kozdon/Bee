import copy
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
from toga.handlers import wrapped_handler
from toga.widgets.base import StyleT
from toga.widgets.canvas import OnResizeHandler, OnTouchHandler
from toga.widgets.canvas.context import Context

from .bases import CustomBackgroundBase, CustomCanvas, CustomContextBase

clr.AddReference("System.Drawing")  # noqa
from System.Drawing import Bitmap as WinBmap  # noqa
from System.Drawing import Image as WinImage  # noqa
from System.Drawing import Size as WinSize  # noqa

PathLikeT: TypeAlias = str | os.PathLike
BytesLikeT: TypeAlias = bytes | bytearray | memoryview
ImageLikeT: TypeAlias = Any
WrappedHandlerT: TypeAlias = Callable[..., object]
ImageContentT: TypeAlias = PathLikeT | BytesLikeT | ImageLikeT


class CVBackground(CustomBackgroundBase):
    def __init__(self, image: toga.Image, id: str, style: StyleT, **kwargs: Any):
        self.image = image
        self.id = id
        self.style = style

    def _draw(self, impl, **kwargs):
        impl.push_context(**kwargs)
        toga.ImageView(self.image, self.id, self.style, kwargs)
        impl.pop_context(**kwargs)


class CVContext(CustomContextBase):
    def __init__(self, canvas: toga.Canvas, **kwargs):
        if not isinstance(canvas, toga.Canvas):
            raise TypeError(
                f"Expected a toga.Canvas instance in the canvas parameter. Got {canvas=}."
            )

        super(Context, CustomContextBase).__init__(self, canvas, kwargs=kwargs)

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
        # Prime the image attribute

        self.height_ = height  # Needed a separate attribute to avoid clashing with height and width coming from style=Pack().
        self.width_ = width  # Needed a separate attribute to avoid clashing with height and width coming from style=Pack().
        self._image: ImageContentT | pathlib.Path | str | None = None
        # print(f"{height, width=}")
        if isinstance(id, str):
            _id = copy.copy(id)

            # id = id.replace("_image", "")

        else:
            raise ValueError(
                f"This app requires id param in the form '<path>_image'. Got {id=}."
            )

        self._id = _id
        # super().__init__(image, id, style, **kwargs)

        if not isinstance(source_path, (str, pathlib.Path)):
            raise TypeError(
                f"The path to the source image is expected to be a str or a pathlib.Path. Got {type(source_path)=}."
            )
        self._source_path = source_path

        if isinstance(image_path, (str, pathlib.Path)):
            self._image_path = image_path

        else:
            _image_path = id.replace("_image", "")

            with pathlib.Path(_image_path) as _path:
                if not _path.exists():
                    raise ValueError(
                        f"The path to the training image could not be retrieved. Provide image_path param or an id param in the form <path_to_the_image.extension>_image. Got {_path=}."
                    )
                if not _path.is_file():
                    raise ValueError(
                        f"The path to the training image could not be retrieved. Provide image_path param or an id param in the form <path_to_the_image.extension>_image. Got {_path=}."
                    )

        toga.Canvas.__init__(
            self,
            id,
            style,
            on_resize=on_resize,
            on_press=on_press,
            on_release=on_release,
            on_drag=on_drag,
            **kwargs,
        )

        toga.ImageView.__init__(self, image, id, style, **kwargs)
        # super().__init__(image=image, id=id, style=style, on_resize=on_resize, on_press=on_press, on_release=on_release, on_drag=on_drag, **kwargs)

        # self.image_view = toga.ImageView(image, id, style, **kwargs)

        self.canvas = toga.Canvas(
            id,
            style,
            on_resize=on_resize,
            on_press=on_press,
            on_release=on_release,
            on_drag=on_drag,
        )

        self.height = height  # Used by Pack() from style kwarg.
        self.width = width  # Used by Pack() from style kwarg.

        # The bases of CustomCanvas base have self._on_resize, etc, already defined and they are not meant to be changed here.
        # Handlers are meant to be saved only as public params.
        self.on_resize = on_resize  # type: ignore
        self.on_press = on_press  # type: ignore
        self.on_release = on_release  # type: ignore
        self.on_drag = on_drag  # type: ignore

        if not isinstance(scaling_factor, (int, float)):
            raise TypeError(
                f"Expected type of the scaling factor is int or float. Got {type(scaling_factor)=}."
            )

        self.scaling_factor = float(scaling_factor)
        self.frame_x: int | None = (
            0  # Raw values before scaling back to the original size of the image.
        )
        self.frame_y: int | None = (
            0  # Raw values before scaling back to the original size of the image.
        )

    def _create(self) -> Any:
        return self.factory.Canvas(interface=self)

    @property
    def image(self) -> toga.Image | None:
        """The image to display.

        When setting an image, you can provide any valid
        [image content][toga.images.ImageContentT] type;
        or [`None`][] to clear the image view.
        """
        if not isinstance(self._image, toga.Image):
            return None

        return self._image

    @image.setter
    def image(self, image: ImageContentT | pathlib.Path | str) -> None:
        if image is None:
            raise ValueError(
                f"No image was provided. Expected ImageContentT | pathlib.Path | str. Got {image=}."
            )
        if isinstance(image, toga.Image):
            self._image = image
        else:
            self._image = toga.Image(image)

        width, height = 0, 0
        print(f"{self.height_=}")

        if isinstance(self.height_, (int, float)):
            height = int(self.height_)

        if isinstance(self.width_, (int, float)):
            width = int(self.width_)

        if pathlib.Path(str(self._image.path)).exists():
            loaded_image = WinImage.FromFile(str(self._image.path))

            if any([not width, not height]):
                height = loaded_image.Height
                width = loaded_image.Width
                print(f"Using original dimensions of the image. {width, height=}.")

            size = WinSize(width, height)
            # print(f"{size=}")
            bitmap = WinBmap(loaded_image, size)
            background = bitmap
            self._impl.native.BackgroundImage = background
        else:
            loaded_image = WinImage.FromFile(str(self._image_path))
            if any([not width, not height]):
                height = loaded_image.Height
                width = loaded_image.Width
                print(f"Using original dimensions of the image. {width, height=}.")

            size = WinSize(width, height)
            # print(f"{size=}")
            bitmap = WinBmap(loaded_image, size)
            background = bitmap
            self._impl.native.BackgroundImage = background

        # area=gtk.Drawingarea()

        # pixbuf=gtk.gdk.pixbuf_new_from_file('background.png')
        # pixmap, mask=pixbuf.render_pixmap_and_mask()

        # area.window.set_back_pixmap(pixmap, False)

        # self._impl.set_image(image)
        self.refresh()

    @property
    def on_press(self) -> OnTouchHandler | Callable | WrappedHandlerT | None:
        """The handler invoked when the canvas is pressed. When a mouse is being used,
        this press will be with the primary (usually the left) mouse button."""

        return self._on_press

    @on_press.setter
    def on_press(self, handler: OnTouchHandler) -> None:
        print(f"Wrapping {handler=}")
        self._on_press = wrapped_handler(self, handler)

    @property
    def on_release(self) -> OnTouchHandler | Callable | WrappedHandlerT | None:
        """The handler invoked when a press on the canvas ends."""
        return self._on_release

    @on_release.setter
    def on_release(self, handler: OnTouchHandler) -> None:
        self._on_release = wrapped_handler(self, handler)

    @property
    def on_drag(self) -> OnTouchHandler | Callable | WrappedHandlerT | None:
        """The handler invoked when the location of a press changes."""
        return self._on_drag

    @on_drag.setter
    def on_drag(self, handler: OnTouchHandler) -> None:
        self._on_drag = wrapped_handler(self, handler)
