import copy
import os
import pathlib
from types import ModuleType
from typing import Generator  # noqa
from typing import OrderedDict  # noqa
from typing import Any, AsyncGenerator, Callable, TypeAlias, Union  # noqa

import clr
import cv2 as cv
import numpy  # noqa: F401
import toga
import toga.handlers
import toga.paths
import toga.sources
import toga.validators
from PIL import Image
from toga.sources.list_source import Row
from toga.style.pack import COLUMN, ROW, Pack
from toga.widgets.button import OnPressHandler
from toga.widgets.canvas import OnTouchHandler
from toga.widgets.multilinetextinput import OnChangeHandler
from toga.widgets.table import OnSelectHandler

from ..widgets.opencv_widgets import CVView
from .decorators import timing
from .models_and_configs import get_models_page

# pylint: disable-next=no-member,unknown-option-value
clr.AddReference("System.Drawing")  # noqa # type: ignore # pylint: disable-next=wrong-import-order,wrong-import-position,import-position,unused-import,import-error
from System.Drawing import Image as WinImage  # noqa # type: ignore

PathLikeT: TypeAlias = str | os.PathLike
BytesLikeT: TypeAlias = bytes | bytearray | memoryview
ImageLikeT: TypeAlias = Any
ImageContentT: TypeAlias = PathLikeT | BytesLikeT | ImageLikeT

# ImageContentT: TypeAlias = Any
# StyleT: TypeAlias = Any

# from diffusers import AutoPipelineForText2Image

# Source - https://stackoverflow.com/a
# Posted by Jason Grout
# Retrieved 2025-11-17, License - CC BY-SA 3.0

startup_libs: list[str] = ["torch", "diffusers"]

diffusers: ModuleType = ModuleType("diffusers")
torch: ModuleType = ModuleType("torch")

default_pipeline: str = "StableDiffusionPipeline"
recommended_base: str = "sd-legacy/stable-diffusion-v1-5"
recommended_lora: str = "latent-consistency/lcm-lora-sdv1-5"

image_dimensions: Callable[..., dict[str, int]] = lambda: {"height": 240, "width": 320}  # noqa: E731

BeeBeeware: type | None = None


@timing
def default(instance: toga.Widget) -> None:
    """
    Docstring for default:
    A callable to function as a dict constructor with default
    values in Field instances.

    :param instance: Description
    :type instance: toga.Widget
    """
    App = instance.app

    if not isinstance(App, toga.App):
        raise TypeError(f"Expected an instance of a toga.App. Got {App=}.")

    if not getattr(App, "config"):
        raise KeyError("The app's instance did not initialize the config attribute.")

    getattr(App, "config")["base_model"] = recommended_base
    getattr(App, "config")["lora_model"] = recommended_lora

    raise NotImplementedError


recommended_config: dict[str, dict[str, Union[str, float, int, bool]]] = {
    "base": {
        "prior_loss_weight": 1.0,
        "resolution": 512,
        "no_half_vae": True,
        "text_encoder_lr": 0.0001,
    },
    "small_dataset": {
        "train_batch_size": 2,
        "learning_rate": 1e-4,
        "max_train_steps": 1500,
        "lr_scheduler": "cosine",
        "lr_warmup_steps": 150,
        "network_dim": 32,
        "network_alpha": 16,
    },
    "medium_dataset": {
        "train_batch_size": 2,
        "learning_rate": 2e-4,
        "max_train_steps": 3000,
        "lr_scheduler": "cosine",
        "lr_warmup_steps": 300,
        "network_dim": 32,
        "network_alpha": 16,
    },
    "big_dataset": {
        "train_batch_size": 2,
        "learning_rate": 2e-4,
        "max_train_steps": 4500,
        "lr_scheduler": "cosine",
        "lr_warmup_steps": 300,
        "network_dim": 64,
        "network_alpha": 32,
    },
}


@timing
def use_recommended(widget: toga.Widget, **kwargs):
    if not isinstance(widget, toga.Button):
        raise TypeError(
            f"The handler was not called by a toga.Button. The handler was called by {widget=}."
        )

    config = copy.copy(recommended_config["base"])

    match widget.id:
        case "small_dataset_button":
            config.update(recommended_config["small_dataset"])
        case "medium_dataset_button":
            config.update(recommended_config["medium_dataset"])
        case "big_dataset_button":
            config.update(recommended_config["big_dataset"])
        case _:
            raise KeyError(
                f"The button's id is expected to be one of: 'small_dataset_button', 'medium_dataset_button', 'big_dataset_button'. Got {widget.id=}."
            )

    print(f"Using recommended {config=}.")
    getattr(widget.app, "config").update(config)
    print(f"Updated config: {getattr(widget.app, "config")=}.")
    # raise NotImplementedError


class select_previews(OnSelectHandler):
    def __init__(self, instance=None, image_ids: list[int] | None = None):
        self.instance = instance
        self.image_ids = image_ids if isinstance(image_ids, list) else []

    def __call__(self, widget: toga.Table, **kwargs):
        headings: list[str] = []
        heading: str

        self.image_ids.clear()

        if not widget.headings:
            raise ValueError(f"Table widget contained no headings. {widget.headings=}")

        for heading in widget.headings:
            headings.append(heading.lower().replace(" ", "_"))

        if widget.selection is None:
            raise ValueError(
                f"The list of image rows should contain only Row entries. Got {widget.selection}"
            )

        image_rows: list[Row] = (
            copy.copy(widget.selection)
            if isinstance(widget.selection, list)
            else [copy.copy(widget.selection)]
        )

        self.image_ids.extend(list(map(widget.data.index, image_rows)))

        print(f"Selected images: {self.image_ids}")


@timing
def get_next(widget: toga.Widget, page: int) -> Union[None, toga.Widget]:
    next_widget: Union[toga.Widget, None] = None

    if not isinstance(widget, toga.Widget):
        raise TypeError(
            f"The widget's type is incorrect. Expected toga.Widget, got {type(widget)=}"
        )

    match widget.id:
        case "base_models":
            next_widget_data = get_models_page(page_num=page)
            next_widget = toga.Table(
                id=widget.id, headings=widget.headings, data=next_widget_data
            )
        case "lora_models":
            next_widget_data = get_models_page(page_num=page, tags=["lora"])
            next_widget = toga.Table(
                id=widget.id, headings=widget.headings, data=next_widget_data
            )

    return next_widget


@timing
def get_previous(widget: toga.Widget, page: int) -> Union[None, toga.Widget]:
    previous_widget: Union[toga.Widget, None] = None

    if not isinstance(widget, toga.Widget):
        raise TypeError(
            f"The widget's type is incorrect. Expected toga.Widget, got {type(widget)=}"
        )

    match widget.id:
        case "base_models":
            previous_widget_data = get_models_page(page_num=page)
            previous_widget = toga.Table(
                id=widget.id, headings=widget.headings, data=previous_widget_data
            )
        case "lora_models":
            previous_widget_data = get_models_page(page_num=page, tags=["lora"])
            previous_widget = toga.Table(
                id=widget.id, headings=widget.headings, data=previous_widget_data
            )

    return previous_widget


class cv_press(OnTouchHandler):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, widget: toga.Canvas, x: int, y: int, **kwargs: Any) -> None:
        print(f"{widget.id=}")
        print(f"{x, y=}")
        print(f"{widget.context}")

        frame_size: dict[str, int] = {
            "height": int(image_dimensions()["height"] * widget.scaling_factor),
            "width": int(image_dimensions()["width"] * widget.scaling_factor),
        }

        if len(widget.context) > 0:
            widget.context.clear()

        with widget.context.Stroke(color="RED", line_width=4.0) as stroke:
            min_y, min_x = 0, 0
            max_y = widget._impl.native.BackgroundImage.Height
            max_x = widget._impl.native.BackgroundImage.Width

            if x + frame_size["width"] >= max_x:
                x = max_x - frame_size["width"]

            if x < min_x:
                x = min_x

            if y + frame_size["height"] >= max_y:
                y = max_y - frame_size["height"]

            if y < min_y:
                y = min_y

            stroke.rect(
                x=x, y=y, width=frame_size["width"], height=frame_size["height"]
            )

        widget.frame_x = x
        widget.frame_y = y

        widget.redraw()


class cv_drag(OnTouchHandler):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, widget: toga.Canvas, x: int, y: int, **kwargs: Any) -> None:
        print(f"{widget.id=}")
        print(f"{x, y=}")

        frame_size: dict[str, int] = {
            "height": int(image_dimensions()["height"] * widget.scaling_factor),
            "width": int(image_dimensions()["width"] * widget.scaling_factor),
        }

        if len(widget.context) > 0:
            widget.context.clear()

        with widget.context.Stroke(color="REBECCAPURPLE", line_width=4.0) as stroke:
            min_y, min_x = 0, 0
            max_y = widget._impl.native.BackgroundImage.Height
            max_x = widget._impl.native.BackgroundImage.Width

            if x + frame_size["width"] >= max_x:
                x = max_x - frame_size["width"]

            if x < min_x:
                x = min_x

            if y + frame_size["height"] >= max_y:
                y = max_y - frame_size["height"]

            if y < min_y:
                y = min_y

            stroke.rect(
                x=x, y=y, width=frame_size["width"], height=frame_size["height"]
            )

        widget.frame_x = x
        widget.frame_y = y

        widget.redraw()


def update_crops(widget: toga.Widget | None = None) -> None:
    if not isinstance(widget, toga.Widget):
        raise TypeError(f"Expected toga.Widget instance. Got {type(widget)=}.")

    crop_and_source_box: toga.Widget | None = widget.parent

    if not isinstance(crop_and_source_box, toga.Widget):
        raise TypeError(
            f"Something went wrong. toga.Box containing images was not found. Got {crop_and_source_box=}."
        )

    content: list[toga.Widget] = crop_and_source_box.children

    img_path = widget._image_path  # noqa: F841
    source_path = widget._source_path  # noqa: F841
    frame_x = widget.frame_x  # noqa: F841
    frame_y = widget.frame_y  # noqa: F841
    scaling_factor = widget.scaling_factor  # noqa: F841
    width, height = image_dimensions()["width"], image_dimensions()["height"]  # noqa: F841

    point1 = frame_x * 1 / scaling_factor, frame_y * 1 / scaling_factor
    point2 = point1[0] + width, point1[1] + height  # noqa: F841

    # img = numpy.asarray(Image.open(source_path))
    # img = img[:, :, ::-1]  # BGR -> RGB colours.
    img = Image.open(source_path).crop((point1[0], point1[1], point2[0], point2[1]))

    # subprocess.check_call(["attrib", "-H", img_path])

    if not pathlib.Path(img_path).exists():
        raise FileExistsError(f"File {img_path=} was not found.")

    if not pathlib.Path(img_path).is_file():
        raise FileExistsError(f"The path {img_path=} does not lead to a file.")

    replacement_img = toga.Image(src=img)
    # replacement_view = toga.ImageView(
    #                     image=replacement_img,
    #                     id=f"{str(img_path)}",
    #                     height=image_dimensions().get("height"),
    #                     width=image_dimensions().get("width"),
    #                 )

    if content:
        if not isinstance(content[0], toga.ImageView):
            raise TypeError(
                f"Expected the first element of the box to be a toga.ImageView instance. Got {content[0]=}."
            )

        # Leftmost widget in the box. Corresponds to the cropped image.

        content[0].image = replacement_img
        crop_and_source_box.refresh()
        img_path_no_extension = str(img_path).split(".")[0]
        extension = str(img_path).split(".")[1]
        content[0].image.save(f"{img_path_no_extension}_cropped.{extension}")
        # content[0].image.save(img_path)

    # raise NotImplementedError


class cv_release(OnTouchHandler):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, widget: toga.Canvas, x: int, y: int, **kwargs: Any) -> None:
        print(f"{widget.id=}")
        print(f"{x, y=}")

        frame_size: dict[str, int] = {
            "height": int(image_dimensions()["height"] * widget.scaling_factor),
            "width": int(image_dimensions()["width"] * widget.scaling_factor),
        }

        if len(widget.context) > 0:
            widget.context.clear()

        with widget.context.Stroke(color="GREEN", line_width=4.0) as stroke:
            min_y, min_x = 0, 0
            max_y = widget._impl.native.BackgroundImage.Height
            max_x = widget._impl.native.BackgroundImage.Width

            if x + frame_size["width"] >= max_x:
                x = max_x - frame_size["width"]

            if x < min_x:
                x = min_x

            if y + frame_size["height"] >= max_y:
                y = max_y - frame_size["height"]

            if y < min_y:
                y = min_y

            stroke.rect(
                x=x, y=y, width=frame_size["width"], height=frame_size["height"]
            )

        widget.frame_x = x
        widget.frame_y = y

        point1 = (
            int(1 / widget.scaling_factor * widget.frame_x),
            int(1 / widget.scaling_factor * widget.frame_y),
        )
        point2 = (
            int(point1[0] + image_dimensions()["width"]),
            int(point1[1] + image_dimensions()["height"]),
        )

        scaled_img_dimensions = (
            widget._impl.native.BackgroundImage.Width,
            widget._impl.native.BackgroundImage.Height,
        )

        print(
            f"Frame cropping points: {point1=} and {point2=}.\nOriginal image dimensions {scaled_img_dimensions=}."
        )

        try:
            update_crops(widget)
        except NotImplementedError as err:
            print(f"{err}. Function update_crops() needs to be implemented.")

        widget.redraw()


class crop_image(OnPressHandler):
    def __init__(
        self,
        window: toga.Window,
        source_id: str,
    ):
        if window is None:
            raise ValueError(f"Expected toga.Window instance. Got {type(window)}.")
        self.window = window

        try:
            if window.widgets[source_id] is None:
                raise ValueError(f"Image with widget id {source_id=} was not found.")

        except KeyError as err:
            raise ValueError(
                f"Image with widget id {source_id=} was not found."
            ) from err

        self.source_id = source_id

    def __call__(self, widget, **kwargs):
        raise NotImplementedError

        image_id = widget.id.replace("_button", "_image")

        if not isinstance(self.window.app, BeeBeeware):
            raise TypeError(
                f"Expected app instance to be of the BeeBeeware type. Got {type(self.window.app)=}."
            )

        dimensions = self.window.app.config.get("image_dimensions")

        if not isinstance(dimensions, dict):
            raise TypeError(
                f"Expected type for config['image__dimensions'] is dict[str, int]. Got {type(dimensions)=}."
            )

        source_path = self.source_id
        source_image = cv.imread(source_path)  # noqa: F841

        if source_image is None:
            raise FileExistsError(
                f"Could not read the {source_image=}. Check if the file's path exists."
            )

        height = dimensions.get("height")  # noqa: F841
        width = dimensions.get("width")  # noqa: F841

        if not isinstance(height, int):
            raise TypeError(f"Expected height type is int. Got {type(height)=}.")

        if not isinstance(width, int):
            raise TypeError(f"Expected height type is int. Got {type(width)=}.")

        starting_point: tuple[int, ...] = tuple([0, 0])

        frame = (
            starting_point,
            (starting_point[0] + width + 2, starting_point[1] + height + 2),
        )
        cv.rectangle(source_image, frame[0], frame[1])  # pylint: disable=no-member

        cropping_window = toga.Window(
            id=f"{image_id}_crop", title=f"Cropping image {self.source_id}"
        )  # pylint: disable=not-callable

        cropping_box = toga.Box(style=Pack(direction=COLUMN))

        cropping_window.content = cropping_box
        cropping_window.show()


class save_captions(OnChangeHandler):
    def __init__(self): ...

    def __call__(self, widget, **kwargs):
        raise NotImplementedError


class confirm_images(OnPressHandler):
    def __init__(
        self,
        window: Union[toga.Window, str, None],
        images_list: list[
            Union[str, pathlib.Path, None]
        ],  # Paths to the copies of the images in the training data folder. Can be overwritten with crops.
        image_ids: list[int],
        source_imgs_list: list[
            Union[str, pathlib.Path, None]
        ],  # Paths to the source images. Do not overwrite the images.
    ):
        if window is None:
            raise ValueError(f"Expected toga.Window instance. Got {type(window)}.")
        self.window = window

        if images_list is None:
            raise ValueError(f"Expected list[str] instance. Got {type(images_list)}.")
        self.images_list = images_list

        if image_ids is None:
            raise ValueError(f"Expected list[int] instance. Got {type(image_ids)}.")
        self.image_ids = image_ids

        if images_list is None:
            raise ValueError(
                f"Expected list[Union[str, pathlib.Path]] instance. Got {type(source_imgs_list)}."
            )
        self.source_imgs_list = source_imgs_list

    def __call__(self, widget, **kwargs) -> None:
        table_id = widget.id.replace("_button", "")  # noqa: F841
        old_view = self.window.widgets[table_id]
        new_view = toga.Box(id=table_id, style=Pack(direction=COLUMN))
        parent_: toga.Box | None = self.window.widgets[table_id].parent

        views_paths = [self.images_list[id] for id in self.image_ids]
        print("Loading previews.")

        if parent_ is None:
            raise ValueError(
                f"Expected parent node of the {widget.id=} to be a toga.Box instance. Got {parent_}."
            )

        for image_id, image in enumerate(views_paths):
            img_path = image
            if not pathlib.Path(img_path).exists():
                raise LookupError(f"The path invalid. Got {img_path=}")
            if not pathlib.Path(img_path).is_file():
                raise FileNotFoundError(
                    f"The path does not lead to a file. Got {img_path=}"
                )

            img = toga.Image(img_path)
            img_height = image_dimensions().get("height")
            if not isinstance(img_height, int):
                raise TypeError(
                    f"Expected image height to be an int. Got {type(img_height)=}"
                )

            scaling_factor = img_height / img.height

            source_img_path: pathlib.Path | str = self.source_imgs_list[image_id]
            if not pathlib.Path(source_img_path).exists():
                raise LookupError(f"The path invalid. Got {source_img_path=}")
            if not pathlib.Path(source_img_path).is_file():
                raise FileNotFoundError(
                    f"The path does not lead to a file. Got {source_img_path=}"
                )
            print(f"{img_path, source_img_path=}")
            source_img: toga.Image = toga.Image(source_img_path)

            crop_press = self.window.app.aux_buttons.get("Crop_image")

            img_view = toga.Box(
                id=str(img_path),
                children=[
                    toga.ImageView(
                        source_img,
                        id=f"{str(source_img_path)}",
                        height=image_dimensions().get("height"),
                        width=image_dimensions().get("width"),
                    ),
                    toga.Button(
                        "Crop Image",
                        id=f"{str(img_path)}_button",
                        on_press=crop_press,
                    ),
                    CVView(
                        image=img,  # Locks the copied images open, thus cannot be overwritten.
                        id=f"{str(img_path)}_image",
                        height=image_dimensions().get("height"),
                        width=int(scaling_factor * img.width),
                        on_press=cv_press(),
                        on_drag=cv_drag(),
                        on_release=cv_release(),  # Has to save as a new file, because the img is in use by CVView.
                        image_path=img_path,
                        source_path=source_img_path,
                        scaling_factor=scaling_factor,
                    ),
                ],
            )

            caption_box = toga.MultilineTextInput(
                id=f"{str(img_path)}_caption",
                placeholder="Caption the image or leave empty to auto-generate captions.",
                on_change=save_captions,
            )

            captioned_img_view = toga.Box(
                id=f"{str(img_path)}_captioned",
                children=[img_view, caption_box],
                style=Pack(direction=COLUMN),
            )

            cv_release_ = cv_release()
            cv_release_(
                img_view.children[2], 0, 0
            )  # img_view.children[2] points to the CVView from toga.Box above.
            update_crops(img_view.children[2])
            new_view.add(captioned_img_view)

        parent_.replace(old_view, new_view)

        # return widget


@timing
def update_selection(
    widget: toga.Widget, selection_list: list[Union[str, pathlib.Path]] | None = None
) -> None:
    if not isinstance(selection_list, list):
        raise TypeError(
            f"Expected selection list to be of the list type. Got {type(selection_list)}."
        )

    window_: toga.Window | toga.MainWindow | None = widget.window
    if not isinstance(window_, (toga.Window, toga.MainWindow)):
        raise TypeError(
            f"Window is expected to be an instance of toga.Window | toga.MainWindow. Got {type(window_)}."
        )

    for index, item in enumerate(selection_list):
        selection_list[index] = str(item)

    window_name = window_.title
    id_ = f"{window_name}_file_name"
    accessors = window_.widgets[id_].items._accessors  # noqa: F841
    print(f"Config files found: {selection_list=}")
    old_selection = window_.widgets[id_]  # type: ignore[arg-type]
    new_selection = toga.Selection(
        id=f"{window_name}_file_name", items=selection_list, style=Pack(direction=ROW)
    )

    window_.widgets[id_].parent.replace(old_selection, new_selection)

    # raise NotImplementedError
