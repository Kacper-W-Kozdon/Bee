"""
An app for Bee.
"""

import asyncio
import contextlib
import copy
import importlib
import importlib.util
import json
import os
import pathlib
import shutil
import sys
import typing
from dataclasses import dataclass, field
from functools import partial, wraps
from io import StringIO
from types import ModuleType
from typing import Generator  # noqa
from typing import OrderedDict  # noqa
from typing import Any, AsyncGenerator, Callable, TypeAlias, Union  # noqa

import clr
import toga
import toga.handlers
import toga.paths
import toga.sources
import toga.validators
from huggingface_hub import hf_hub_download
from toga.colors import rgb
from toga.constants import Baseline
from toga.fonts import SANS_SERIF
from toga.style.pack import CENTER, COLUMN, ROW, Pack
from toga.widgets import textinput
from toga.widgets.canvas import OnTouchHandler
from toga.widgets.table import OnSelectHandler

from .auxiliary.helpers.decorators import timing  # noqa
from .auxiliary.helpers.handle_files import list_files  # noqa
from .auxiliary.helpers.handlers import crop_image  # noqa
from .auxiliary.helpers.handlers import get_next  # noqa
from .auxiliary.helpers.handlers import get_previous  # noqa
from .auxiliary.helpers.handlers import update_selection  # noqa
from .auxiliary.helpers.handlers import confirm_images, select_previews  # noqa
from .auxiliary.helpers.models_and_configs import get_models  # noqa
from .auxiliary.helpers.models_and_configs import get_models_page  # noqa
from .auxiliary.helpers.models_and_configs import update_config  # noqa
from .auxiliary.widgets.opencv_widgets import CVContext  # noqa

from .auxiliary.helpers.models_and_configs import get_default_base_and_lora  # isort: skip

clr.AddReference("System.Drawing")  # noqa
from System.Drawing import Image as WinImage  # noqa

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


class Loader:
    def __init__(self, loadable: str):
        self.loadable = loadable

    def __await__(self):
        lib = self.loadable
        print(f"Loading {lib}")
        if lib not in sys.modules:
            globals().update({lib: importlib.import_module(lib)})

        return (yield None)


async def loader(libraries: list[str], counter: int = 0):
    for lib in libraries:
        print(f"{lib=}, {lib in sys.modules=}")

        if lib not in sys.modules:
            await Loader(lib)
        counter += 1

        print(f"{lib in sys.modules=}")
        yield counter


async def load_libs(libraries: list[str], widget: toga.Widget):
    async for item in loader(libraries):
        widget.value = item

        print(widget.value)

        await asyncio.sleep(0.1)


class canvas_press(OnTouchHandler):
    def __init__(self):
        pass

    def __call__(self, widget, x, y, **kwargs):
        raise NotImplementedError


class canvas_drag(OnTouchHandler):
    def __init__(self):
        pass

    def __call__(self, widget, x, y, **kwargs):
        raise NotImplementedError


class canvas_release(OnTouchHandler):
    def __init__(self):
        pass

    def __call__(self, widget, x, y, **kwargs):
        raise NotImplementedError


@contextlib.contextmanager
def capture():
    oldout, olderr = sys.stdout, sys.stderr
    try:
        out = [StringIO(), StringIO()]
        sys.stdout, sys.stderr = out
        yield out
    finally:
        sys.stdout, sys.stderr = oldout, olderr
        out[0] = out[0].getvalue()
        out[1] = out[1].getvalue()


# with capture() as out:
#     pass

# print(out)


# def lazy(fullname):
#     try:
#         return sys.modules[fullname]
#     except KeyError:
#         spec = importlib.util.find_spec(fullname)
#         module = importlib.util.module_from_spec(spec)
#         loader = importlib.util.LazyLoader(spec.loader)
#         # Make module with proper locking and get it inserted into sys.modules.
#         loader.exec_module(module)
#         return module


# diffusers = lazy("diffusers")
# torch = lazy("torch")


@timing
def assign_container(fun):
    # print(f"{fun.__name__=}")

    @wraps(fun)
    def outer(instance, container=None, page_id=None):
        # print(f"{container=}")
        # print(f"{instance=}")

        def inner(widget, instance=instance):
            return fun(instance, widget, container=container, page_id=page_id)

        return inner

    return outer


no_preview_list: Callable[..., list[str]] = lambda: list(  # noqa: E731
    [
        "no_preview",
        "config_path",
        "base_model",
        "lora_model",
        "pipeline",
        "Save_path",
        "Load_path",
        "source_path",
        "image_dimensions",
    ]
)

image_dimensions: Callable[..., dict[str, int]] = lambda: {"height": 240, "width": 320}  # noqa: E731


@timing
def default(instance: toga.Widget) -> None:
    instance.config["base_model"] = ""
    instance.config["lora_model"] = ""


@timing
async def train_model(
    instance: toga.Widget,
) -> Union[None, AsyncGenerator[StringIO, Any]]:
    with capture() as out:
        if "diffusers" not in sys.modules:
            diffusers = importlib.import_module("diffusers")

        try:
            diffusers.__name__ in sys.modules
        except ValueError:
            diffusers = sys.modules["diffusers"]

        DiffusionPipeline = diffusers.DiffusionPipeline
        DDIMScheduler = diffusers.DDIMScheduler

        # Source: https://huggingface.co/ByteDance/Hyper-SD

        base_model_id = "runwayml/stable-diffusion-v1-5"
        repo_name = "ByteDance/Hyper-SD"
        # Take 2-steps lora as an example
        ckpt_name = "Hyper-SD15-2steps-lora.safetensors"
        # Load model.
        pipe = DiffusionPipeline.from_pretrained(
            base_model_id, torch_dtype=torch.float16, variant="fp16"
        ).to("cuda")
        pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
        pipe.fuse_lora()
        # Ensure ddim scheduler timestep spacing set as trailing !!!
        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )
        prompt = "a photo of a cat"
        image = pipe(prompt=prompt, num_inference_steps=2, guidance_scale=0).images[0]  # noqa: F841

    yield out[0]
    raise NotImplementedError


@dataclass
class Config:
    no_preview: list[str] = field(default_factory=no_preview_list)
    config_path: Union[str, pathlib.Path] = ""
    Save_path: Union[str, pathlib.Path] = ""
    Load_path: Union[str, pathlib.Path] = ""
    source_path: Union[str, pathlib.Path] = ""
    image_dimensions: dict[str, int] = field(default_factory=image_dimensions)
    base_model: Union[str, None] = ""
    lora_model: Union[str, None] = ""
    pipeline: Union[str, None] = default_pipeline


main_config = Config()


class BeeBeeware(toga.App):
    placeholder_text = "Placeholder"
    main_window_split = {"menu": 1, "previews": 2}
    preview_container_split = {"menu": 1, "options": 1}
    config: OrderedDict[
        str, Union[str, pathlib.Path, dict[str, int], list[str], None]
    ] = OrderedDict(
        {
            config_name: config_value
            for config_name, config_value in main_config.__dict__.items()
        }
    )
    files_list = []
    images_list = []

    def startup(self):
        """Construct and show the Toga application.

        Usually, you would add your application to a main content box.
        We then create a main window (with a name matching the app), and
        show the main window.
        """

        self.config.update({"config_path": str(toga.paths.Paths().config)})
        self.config.update(
            {"base_model": get_default_base_and_lora("text-to-image", ["lora"])[0]}
        )
        self.config.update(
            {"lora_model": get_default_base_and_lora("text-to-image", ["lora"])[1]}
        )

        self.main_window = toga.MainWindow(title=self.formal_name)

        loading_progress = toga.ProgressBar(
            "loading_progress",
            max=len(startup_libs),
            value=0,
        )

        loading_progress.start()
        asyncio.ensure_future(
            load_libs(startup_libs, loading_progress), loop=asyncio.get_running_loop()
        )
        loading_progress.stop()

        self.main_buttons = OrderedDict(
            {
                "Model": self.preview_model_menu,
                "Config": self.preview_config,
                "Training Images": self.preview_images,
                "Summary": self.preview_summary,
            }
        )

        self.aux_buttons = OrderedDict(
            {
                "Load from file": self.load_config,
                "Save to file": self.save_config,
                "Next": self.next,
                "Previous": self.previous,
                "Default": default,
                "Train": train_model,
                "confirm_images": confirm_images,
                "Crop_image": crop_image,
            }
        )

        menu_previews_split = toga.SplitContainer(style=Pack(direction=COLUMN))

        menu = toga.Box(id="menu", style=Pack(direction=COLUMN, alignment=CENTER))

        for button_name, button_action in self.main_buttons.items():
            menu.add(
                toga.Button(
                    f"{button_name}",
                    on_press=button_action,
                    style=Pack(width=200, margin=20),
                )
            )

        previews = toga.Box(style=Pack(direction=COLUMN))
        self.previews_container = toga.ScrollContainer(
            id="previews_container", horizontal=False, style=Pack(direction=COLUMN)
        )
        previews.add(loading_progress)
        self.previews_container.content = previews

        menu_previews_split.content = [
            (menu, self.main_window_split["menu"]),
            (self.previews_container, self.main_window_split["previews"]),
        ]

        # main_box.add(menu_previews_split)

        self.canvas = toga.Canvas(
            style=Pack(flex=1, direction=ROW),
            on_resize=self.on_resize,
            on_press=self.on_press,
            alignment=CENTER,
        )

        self.main_window.content = menu_previews_split
        self.main_window.show()

    def preview_model_menu(self, widget) -> None:
        base_models_data = get_models_page(page_num=1)
        lora_models_data = get_models_page(page_num=1, tags=["lora"])

        class Base_Select(OnSelectHandler):
            def __init__(self, config: OrderedDict[str, Union[str, None]]):
                self.config = config
                super().__init__()

            def __call__(self, widget, **kwargs) -> None:
                base = copy.copy(widget.selection.base_models)
                widget.window.widgets["base_picked"].value = base
                self.config.update({"base_model": base})

        class Lora_Select(OnSelectHandler):
            def __init__(self, config: OrderedDict[str, Union[str, None]]):
                self.config = config
                super().__init__()

            def __call__(self, widget, **kwargs) -> None:
                lora = copy.copy(widget.selection.trainable_model)
                widget.window.widgets["lora_picked"].value = lora
                self.config.update({"lora_model": lora})

        lora_select = Lora_Select(self.config)
        base_select = Base_Select(self.config)

        base_page = toga.Label("1", id="base_page")
        base_models_table = toga.Table(
            id="base_models",
            headings=["Base models"],
            data=base_models_data,
            on_select=base_select,
        )
        base_models_next = toga.Button(
            text="Next",
            on_press=self.aux_buttons["Next"](
                container=base_models_table, page_id=base_page.id
            ),
        )
        base_models_prev = toga.Button(
            text="Previous",
            on_press=self.aux_buttons["Previous"](
                container=base_models_table, page_id=base_page.id
            ),
        )
        base_buttons = toga.Box(
            id="base_buttons",
            style=Pack(direction=ROW),
            children=[base_models_prev, base_page, base_models_next],
        )
        base_picked_label = toga.Label("Base:")
        base_picked_model = toga.TextInput(
            id="base_picked", readonly=True, value=self.config["base_model"]
        )
        base_picked = toga.Box(
            style=Pack(direction=ROW), children=[base_picked_label, base_picked_model]
        )
        base_models_box = toga.Box(
            id="base_box",
            style=Pack(direction=COLUMN),
            children=[base_models_table, base_buttons, base_picked],
        )

        lora_page = toga.Label("1", id="lora_page")
        lora_models_table = toga.Table(
            id="lora_models",
            headings=["Trainable model"],
            data=lora_models_data,
            on_select=lora_select,
        )
        lora_models_next = toga.Button(
            text="Next",
            on_press=self.aux_buttons["Next"](
                container=lora_models_table, page_id=lora_page.id
            ),
        )
        lora_models_prev = toga.Button(
            text="Previous",
            on_press=self.aux_buttons["Previous"](
                container=lora_models_table, page_id=lora_page.id
            ),
        )
        lora_buttons = toga.Box(
            id="lora_buttons",
            style=Pack(direction=ROW),
            children=[lora_models_prev, lora_page, lora_models_next],
        )
        lora_picked_label = toga.Label("Lora:")
        lora_picked_model = toga.TextInput(
            id="lora_picked", readonly=True, value=self.config["lora_model"]
        )
        lora_picked = toga.Box(
            style=Pack(direction=ROW), children=[lora_picked_label, lora_picked_model]
        )
        lora_models_box = toga.Box(
            id="lora_box",
            style=Pack(direction=COLUMN),
            children=[lora_models_table, lora_buttons, lora_picked],
        )

        models = toga.SplitContainer(id="models", style=Pack(direction=COLUMN))
        models.content = [
            (base_models_box, self.preview_container_split["menu"]),
            (lora_models_box, self.preview_container_split["options"]),
        ]
        self.previews_container.content = models

    def preview_images(self, widget) -> None:
        valid_extensions = [".jpg", ".json", ".png", ".jpeg"]

        class check_path_confirm(textinput.OnConfirmHandler):
            def __init__(
                self,
                instance: BeeBeeware,
                files: list[Union[str, pathlib.Path, None]],
                images: list[Union[str, pathlib.Path, None]],
                destination: Union[str, pathlib.Path, None] = None,
                format_: str = "Path",
                extensions: Union[list[str], None] = None,
            ):
                self.format_ = format_
                self.extensions = (
                    [".json"] if not isinstance(extensions, list) else extensions
                )
                self.files = files  # Paths to the source images.
                self.img_previews = images  # Paths to the copies of the images. Can be overwritten with edits/crops.
                self.destination = (
                    pathlib.Path(destination)
                    if destination is not None
                    else pathlib.Path(f"{toga.paths.Paths().config}\\Bee_training_data")
                )
                self.instance = instance

            def __call__(self, widget: toga.TextInput, **kwargs) -> None:
                path: pathlib.Path = pathlib.Path(widget.value)
                destination_path: pathlib.Path = self.destination

                print(f"{path, destination_path=}")

                if not path.exists():
                    confirmation = toga.ConfirmDialog(
                        "Create a folder",
                        f"Do you wish to create the folder with the {path=}?",
                    )
                    if confirmation:
                        path.mkdir()
                    else:
                        return None

                if not destination_path.exists():
                    confirmation = toga.ConfirmDialog(
                        "Create a folder",
                        f"Do you wish to create the folder with the {destination_path=}?",
                    )
                    if confirmation:
                        destination_path.mkdir()
                    else:
                        return None

                files_to_clear = list_files(destination_path, extension="")

                for file in files_to_clear:
                    if "Bee_training_data" not in str(file):
                        raise FileNotFoundError(
                            f"Expected 'Bee_training_data' folder name was not found in the path to the {file=}."
                        )

                    os.remove(file)

                files_: list[Union[str, pathlib.Path]] = []

                for extension in self.extensions:
                    files_.extend(
                        list_files(  # noqa: F841
                            path, format_=self.format_, extension=extension
                        )
                    )

                images_list_ = []

                files_.sort()

                for file_index, file in enumerate(files_):
                    extension = [ext for ext in valid_extensions if ext in str(file)][0]
                    shutil.copyfile(
                        file,
                        pathlib.Path(f"{destination_path}\\{file_index}{extension}"),
                    )
                    images_list_.append(
                        pathlib.Path(f"{destination_path}\\{file_index}{extension}")
                    )

                # for path in files_:
                #     my_image = None
                #     view = None

                #     if pathlib.Path(path).exists():
                #         my_image = toga.Image(pathlib.Path(path))
                #         view = toga.ImageView(my_image)

                #     images_list_.append(view)

                if len(self.files) != len(self.img_previews):
                    raise IndexError(
                        f"Mismatched lists of file paths and previews. {len(self.files)=}, {len(self.img_previews)=}."
                    )

                while self.files:
                    self.files.clear()
                    self.img_previews.clear()

                self.files.extend(files_)
                self.img_previews.extend(images_list_)

                print(f"Source paths: {self.files=}.")
                print(f"Destination paths: {self.img_previews=}.")

                container_id: str = "files_table"
                old_view: toga.Table = self.instance.main_window.widgets[container_id]
                next_view: toga.Table = toga.Table(
                    id=container_id,
                    data=zip(files_, images_list_),
                    headings=old_view.headings,
                    style=Pack(direction=COLUMN),
                )

                self.instance.main_window.widgets[container_id].parent.replace(
                    old_view, next_view
                )

                return None

        class check_path_change(textinput.OnChangeHandler):
            def __init__(
                self,
                instance: BeeBeeware,
                files: list[Union[str, pathlib.Path, None]],
                images: list[Union[str, pathlib.Path, None]],
                destination: Union[str, pathlib.Path, None] = None,
                format_: str = "Path",
                extensions: Union[list[str], None] = None,
            ):
                self.format_ = format_
                self.extensions = (
                    [".json"] if not isinstance(extensions, list) else extensions
                )
                self.files = files  # Paths to the source images.
                self.img_previews = images  # Paths to the copies of the images. Can be overwritten with edits/crops.
                self.destination = (
                    pathlib.Path(destination)
                    if destination is not None
                    else pathlib.Path(f"{toga.paths.Paths().config}\\Bee_training_data")
                )
                self.instance = instance

            def __call__(self, widget: toga.TextInput, **kwargs) -> None:
                path: pathlib.Path = pathlib.Path(widget.value)
                destination_path: pathlib.Path = self.destination

                if not path.exists():
                    confirmation = toga.ConfirmDialog(
                        "Create a folder",
                        f"Do you wish to create the folder with the {path=}?",
                    )
                    if confirmation:
                        path.mkdir()
                    else:
                        return None

                if not destination_path.exists():
                    confirmation = toga.ConfirmDialog(
                        "Create a folder",
                        f"Do you wish to create the folder with the {destination_path=}?",
                    )
                    if confirmation:
                        destination_path.mkdir()
                    else:
                        return None

                files_to_clear = list_files(destination_path, extension="")

                for file in files_to_clear:
                    if "Bee_training_data" not in str(file):
                        raise FileNotFoundError(
                            f"Expected 'Bee_training_data' folder name was not found in the path to the {file=}."
                        )

                    os.remove(file)

                files_: list[Union[str, pathlib.Path]] = []

                for extension in self.extensions:
                    files_.extend(
                        list_files(  # noqa: F841
                            path, format_=self.format_, extension=extension
                        )
                    )

                images_list_ = []

                files_.sort()

                for file_index, file in enumerate(files_):
                    extension = [ext for ext in valid_extensions if ext in str(file)][0]
                    shutil.copyfile(
                        file,
                        pathlib.Path(f"{destination_path}\\{file_index}{extension}"),
                    )
                    images_list_.append(
                        pathlib.Path(f"{destination_path}\\{file_index}{extension}")
                    )

                # for path in files_:
                #     my_image = None
                #     view = None

                #     if pathlib.Path(path).exists():
                #         my_image = toga.Image(pathlib.Path(path))
                #         view = toga.ImageView(my_image)

                #     images_list_.append(view)

                if len(self.files) != len(self.img_previews):
                    raise IndexError(
                        f"Mismatched lists of file paths and previews. {len(self.files)=}, {len(self.img_previews)=}."
                    )

                while self.files:
                    self.files.clear()
                    self.img_previews.clear()

                self.files.extend(files_)
                self.img_previews.extend(images_list_)

                print(f"Source paths: {self.files=}.")
                print(f"Destination paths: {self.img_previews=}.")

                container_id: str = "files_table"
                old_view: toga.Table = self.instance.main_window.widgets[container_id]
                next_view: toga.Table = toga.Table(
                    id=container_id,
                    data=zip(files_, images_list_),
                    headings=old_view.headings,
                    style=Pack(direction=COLUMN),
                    multiple_select=True,
                    on_select=select_previews(self.instance, image_ids),
                )

                self.instance.main_window.widgets[container_id].parent.replace(
                    old_view, next_view
                )

                # return widget

        images_path: pathlib.Path = pathlib.Path(
            f"{toga.paths.Paths().config}\\Bee_training_data"
        )

        if not images_path.exists():
            images_path.mkdir()

        files_list: list[Union[str, pathlib.Path, None]] = (
            self.files_list
        )  # Paths to the source images.
        images_list: list[Union[None, str, pathlib.Path]] = (
            self.images_list
        )  # Paths to the copies of the images. Can be overwritten with edits/crops.

        selected_path = toga.TextInput(  # noqa: F841
            id="source_path",
            style=Pack(direction=COLUMN),
            on_confirm=check_path_confirm(
                self,
                files_list,
                images_list,
                destination=images_path,
                extensions=[".png", ".jpg", ".jpeg"],
            ),
            on_change=check_path_change(
                self,
                files_list,
                images_list,
                destination=images_path,
                extensions=[".png", ".jpg", ".jpeg"],
            ),
            # readonly=True,
            value=self.config["source_path"],
        )

        select_path = toga.Button(
            "Select path", id="source_path_button", on_press=self.path_handler
        )

        for path in files_list:
            my_image = None
            view = None

            if pathlib.Path(path).exists():
                my_image = toga.Image(pathlib.Path(path))
                view = toga.ImageView(
                    my_image,
                    height=image_dimensions().get("height"),
                    width=image_dimensions().get("width"),
                )

            images_list.append(view)

        image_ids: list[int] = []

        files_table = toga.Table(
            id="files_table",
            style=Pack(direction=COLUMN),
            headings=["Source Images", "Training Previews"],
            data=zip(files_list, images_list),  # zip(source_paths, training_data_paths)
            on_select=select_previews(self, image_ids),
            multiple_select=True,
        )

        confirmed_images = toga.Box(id="confirm_images")  # noqa: F841

        confirm_images_press = self.aux_buttons["confirm_images"]

        if not issubclass(confirm_images_press, confirm_images):
            raise TypeError(
                f"Expected handler confirm_images. Got {confirm_images_press=}."
            )

        confirm_images_ = toga.Button(  # noqa: F841
            "Confirm selection",
            id="confirm_images_button",
            on_press=confirm_images_press(
                window=self.main_window,
                images_list=images_list,  # Training data paths.
                image_ids=image_ids,
                source_imgs_list=files_list,  # Source images paths.
            ),
        )

        selection_box_paths = toga.Box(
            style=Pack(direction=ROW),
            children=[selected_path, select_path],
        )

        selection_box = toga.Box(
            style=Pack(direction=COLUMN),
            children=[
                selection_box_paths,
                files_table,
                confirm_images_,
                confirmed_images,
            ],
        )

        self.previews_container.content = selection_box

        # raise NotImplementedError

    def preview_config(self, widget) -> None:
        if not all([lib in sys.modules for lib in startup_libs]):
            toga.InfoDialog(
                "Please, wait.", "Not all the libraries have been loaded yet."
            )
            pass
            return None

        config = OrderedDict(
            {
                config_name: toga.NumberInput(min=0, max=10, step=0.1, value=1)
                for config_name, _ in self.config.items()
                if config_name not in self.config["no_preview"]
            }
        )

        base_model = self.config.get("base_model")
        # lora_model = self.config.get("lora_model")

        model_configs: OrderedDict[str, Union[str, int, float, list, dict, None]] = (
            update_config(instance=self, model_id=base_model)
        )
        config.update(model_configs)  # type: ignore

        config_scroll = toga.Box(id="config", style=Pack(direction=COLUMN))
        save_button = toga.Button(
            "Save to file", on_press=self.aux_buttons["Save to file"]
        )
        load_button = toga.Button(
            "Load from file", on_press=self.aux_buttons["Load from file"]
        )
        default_button = toga.Button(
            "Use default", on_press=self.aux_buttons["Default"]
        )

        for config_name, config_input in config.items():
            validators: (
                list[
                    Union[
                        toga.validators.CountValidator, toga.validators.BooleanValidator
                    ]
                ]
                | None
            ) = []
            label = toga.Label(config_name)
            types_id = f"type_{label}"
            values_id = f"value_{label}"
            config_id = f"{label}_config"
            input_types = config_input[0]
            input_default = config_input[1]
            add_next: bool = False

            if any([int in input_types, float in input_types]):
                validators.append(toga.validators.Number)

            if any(
                [
                    list in map(typing.get_origin, input_types),
                    dict in map(typing.get_origin, input_types),
                ]
            ):
                add_next = True  # noqa: F841

            config_types = toga.TextInput(id=types_id, value=input_types, readonly=True)
            config_box = toga.Box(id=config_id, style=Pack(direction=ROW))
            config_value = toga.TextInput(
                id=values_id, placeholder=input_default, validators=validators
            )

            config_box.add(label)
            config_box.add(config_types)
            config_box.add(config_value)

            config_scroll.add(config_box)

        save_load_box = toga.Box(id="save_and_load", style=Pack(direction=ROW))

        save_load_box.add(default_button)
        save_load_box.add(load_button)
        save_load_box.add(save_button)

        config_scroll.add(save_load_box)

        self.previews_container.content = config_scroll

    def preview_summary(self, widget) -> None:
        if not all([lib in sys.modules for lib in startup_libs]):
            toga.InfoDialog(
                "Please, wait.", "Not all the libraries have been loaded yet."
            )
            pass
            return None

        summary_preview = toga.Box(id="summary_preview", style=Pack(direction=COLUMN))
        update_config(self)
        for config_label, config_value_ in self.config.items():
            if config_label == "no_preview":
                continue

            if isinstance(config_value_, tuple):
                config_value = config_value_[1]
            else:
                config_value = config_value_

            label = toga.Label(config_label)
            value = toga.TextInput(
                readonly=True, value=config_value, style=Pack(direction=COLUMN)
            )
            config_box = toga.Box(style=Pack(direction=ROW), children=[label, value])
            summary_preview.add(config_box)

        train_button = toga.Button(
            "Train the model", on_press=self.aux_buttons["Train"]
        )
        summary_preview.add(train_button)

        self.previews_container.content = summary_preview

    @assign_container
    def next(
        self,
        widget,
        container: Union[toga.Box, toga.Table, None] = None,
        page_id: Union[None, str] = None,
    ) -> Union[toga.Box, toga.Table, None]:
        # print(f"{container.id=}, {container.id in self.main_window.content.children=}")
        print(f"{self.main_window.widgets[container.id].parent=}")
        # print(f"{self.main_window.widgets[container.id].data=}")
        page = max(int(self.main_window.widgets[page_id].text) + 1, 1)

        # next_view = toga.Table(headings=headings, data=data)

        old_view = self.main_window.widgets[container.id]
        next_view = get_next(old_view, page)
        if next_view is None:
            self.main_window.dialog(
                toga.InfoDialog("Error", "Could not retrieve the next view.")
            )
            raise TypeError(
                f"Next view is expected to be of the type toga.Widget. Got {type(next_view)}"
            )

        self.main_window.widgets[container.id].parent.replace(old_view, next_view)
        # for item in data:
        #     self.main_window.widgets[container.id].data.append(item)

        self.main_window.widgets[page_id].text = str(int(page))
        # print(f"{dir(self.main_window.widgets)=}")
        self.main_window.show()
        return next_view

    @assign_container
    def previous(
        self,
        widget,
        container: Union[toga.Box, toga.Table, None] = None,
        page_id: [None, str] = None,
    ) -> Union[toga.Box, toga.Table, None]:
        # print(f"{container.id=}, {container.id in self.main_window.content.children=}")
        # print(f"{self.main_window.widgets[container.id]=}")
        # print(f"{self.main_window.widgets[container.id].data=}")
        page = max(int(self.main_window.widgets[page_id].text) - 1, 1)

        # prev_view = toga.Table(headings=headings, data=data)

        old_view = self.main_window.widgets[container.id]
        prev_view = get_previous(old_view, page)
        if prev_view is None:
            self.main_window.dialog(
                toga.InfoDialog("Error", "Could not retrieve the previous view.")
            )
            raise TypeError(
                f"Next view is expected to be of the type toga.Widget. Got {type(prev_view)}"
            )

        self.main_window.widgets[container.id].parent.replace(old_view, prev_view)
        # for item in data:
        #     self.main_window.widgets[container.id].data.append(item)

        self.main_window.widgets[page_id].text = str(int(page))
        # print(f"{dir(self.main_window.widgets)=}")
        self.main_window.show()
        return prev_view

    def text_input(self, widget, window_name: str = "") -> toga.Window:
        default: OrderedDict[str, str] = OrderedDict(
            {
                "Save": f"{toga.paths.Paths().config}\\config",
                "Load": f"{toga.paths.Paths().config}\\config",
            }
        )

        for _, path in default.items():
            if not pathlib.Path(path).exists():
                pathlib.Path(path).mkdir()

        text_window = toga.Window(title=window_name)

        config_files: list[str | pathlib.Path] = []

        select_path = toga.Button(
            f"{window_name} path",
            id=f"{window_name}_path_button",
            on_press=partial(self.path_handler, input_list=config_files),
        )
        selected_path = toga.TextInput(
            id=f"{window_name}_path",
            readonly=True,
            value=self.config[f"{window_name}_path"],
            on_change=partial(update_selection, selection_list=config_files),
        )

        entry_box = toga.Box(id=window_name, style=Pack(direction=ROW))
        # text_input_box = toga.TextInput(placeholder=f"{window_name} file name")
        selection_input_box = toga.Selection(
            id=f"{window_name}_file_name", items=config_files, style=Pack(direction=ROW)
        )
        path: str = self.config[f"{window_name}_path"] or default[window_name]

        confirm_button = toga.Button(
            "Confirm",
            on_press=partial(
                self.close_window,
                window=text_window,
                path=path,
                text_input_box=selection_input_box,
            ),
        )

        entry_box.add(selection_input_box)
        entry_box.add(confirm_button)

        path_box = toga.Box(
            style=Pack(direction=ROW), children=[selected_path, select_path]
        )

        text_window.content = toga.Box(
            style=Pack(direction=COLUMN), children=[path_box, entry_box]
        )
        self.text_window = text_window
        text_window.show()

        return text_window

    async def load_config(self, widget) -> None:
        self.text_input(widget, window_name="Load")

    async def save_config(self, widget) -> None:
        self.text_input(widget, window_name="Save")

    def close_window(
        self,
        widget,
        window: toga.Window,
        path: str,
        text_input_box: toga.TextInput,
        config: Config = Config(),
    ) -> Config:
        print(f"Closing {window.title=}")

        if text_input_box.value:
            config_name = f"{text_input_box.value}.json"

        else:
            config_name = "beeconfig.json"

        save_load_path = self.text_window.widgets[f"{window.title}_path"].value

        path = save_load_path or path
        config_path: Union[str, pathlib.Path] = ""
        if window.title == "Load":
            if (config.config_path == "") or (path is not None):
                config.config_path = path

            config_path = config.config_path

            config_path = pathlib.Path(config_path)

            if not config_path.exists() and (config_path is not None):
                print(f"Creating {config_path=}")
                pathlib.Path(f"{config_path}\\").mkdir()
                model_config = self.config

                for option_name, option_value in model_config.items():
                    if isinstance(option_value, tuple):
                        model_config[option_name] = option_value[1]

                json_configs = json.dumps(model_config)

                with open(
                    pathlib.Path(f"{config_path}\\{config_name}"), "+w"
                ) as config_file:
                    config_file.write(json_configs)

            if config_path is not None:
                config_dict: OrderedDict = OrderedDict({})
                with open(
                    pathlib.Path(f"{config_path}\\{config_name}"), "r"
                ) as config_file:
                    config_dict = OrderedDict(json.load(config_file))

                for option_name, option_value in config_dict.items():
                    setattr(config, option_name, option_value)

        if window.title == "Save":
            if path is not None:
                config.config_path = path

            config_path = config.config_path

            config_path = pathlib.Path(config_path)

            if not config_path.exists() and (config_path is not None):
                print(f"Creating {config_path=}")
                pathlib.Path(f"{config_path}\\").mkdir()
                model_config = self.config
                json_configs = json.dumps(model_config)

                with open(
                    pathlib.Path(f"{config_path}\\{config_name}"), "+w"
                ) as config_file:
                    config_file.write(json_configs)

            if config_path is not None:
                config_to_save = copy.copy(config.__dict__)
                config_to_save["config_path"] = str(config.__dict__["config_path"])
                json_config = json.dumps(config_to_save)

                with open(
                    pathlib.Path(f"{config_path}\\beeconfig.json"), "+w"
                ) as config_file:
                    config_file.write(json_config)

                print(f"Config file saved at {config_path=}")

        window.close()
        return config

    def draw_text(self, widget):
        print("Writing on canvas.")
        self.previews_container.content = self.canvas
        font = toga.Font(family=SANS_SERIF, size=20)
        self.text_width, text_height = self.canvas.measure_text(
            self.placeholder_text, font
        )

        # print(f"{self.main_window.size.width=}")

        x = (150 - self.text_width) // 2
        y = 10

        self.canvas.context.clear()

        with self.canvas.Stroke(color="REBECCAPURPLE", line_width=4.0) as rect_stroker:
            self.text_border = rect_stroker.rect(
                x - 5,
                y - 5,
                self.text_width + 10,
                text_height + 10,
            )
        with self.canvas.Fill(color=rgb(149, 119, 73)) as text_filler:
            self.text = text_filler.write_text(
                self.placeholder_text, x, y, font, Baseline.TOP
            )

        # self.previews_container.content.redraw()

    def on_resize(self, widget, width, height, **kwargs):
        # On resize, center the text horizontally on the canvas. on_resize will be
        # called when the canvas is initially created, when the drawing objects won't
        # exist yet. Only attempt to reposition the text if there's context objects on
        # the canvas.
        if widget.context:
            left_pad = (width - self.text_width) // 2
            self.text.x = left_pad
            self.text_border.x = left_pad - 5
            widget.redraw()

    async def on_press(self, widget, x, y, **kwargs):
        await self.main_window.dialog(
            toga.InfoDialog("Placeholder title", "Placeholder message.")
        )

    def path_handler(
        self, widget, input_list: list[Union[str, pathlib.Path]] | None = None, **kwargs
    ):
        if isinstance(input_list, list):
            input_list.clear()
        images_path: pathlib.Path = pathlib.Path(
            f"{toga.paths.Paths().config}\\Bee_training_data"
        )

        save_load_path: pathlib.Path = pathlib.Path(
            f"{toga.paths.Paths().config}\\config"
        )

        if not images_path.exists():
            images_path.mkdir()

        captions_paths_dict = {
            "Save_path_button": ("Select config save folder", save_load_path),
            "Load_path_button": ("Select config load folder", save_load_path),
            "source_path_button": ("Select training data folder", images_path),
        }

        source_path = toga.SelectFolderDialog(
            captions_paths_dict[widget.id][0],
            initial_directory=captions_paths_dict[widget.id][1],
        )

        task_name = str(widget.id).replace("_button", "")

        task = asyncio.create_task(self.main_window.dialog(source_path), name=task_name)
        task.add_done_callback(partial(self.dialog_dismissed, input_list=input_list))
        print("Dialog has been created")

    def dialog_dismissed(
        self, task, input_list: list[Union[str, pathlib.Path]] | None = None
    ):
        widget_name: str = task.get_name()
        if isinstance(input_list, list):
            input_list.clear()

        if task.result():
            print(f"{task.result()=}")

            if isinstance(input_list, list):
                files = list_files(mypath=task.result())
                input_list.extend(files)
                print(f"Files found: {input_list=}.")

            match widget_name:
                case "Save_path" | "Load_path":
                    self.text_window.widgets[widget_name].value = task.result()
                    self.config[widget_name] = task.result()

                case "source_path":
                    self.main_window.widgets[widget_name].value = task.result()
                    self.config[widget_name] = task.result()

                case _:
                    raise KeyError(
                        f"Couldn't match the path with the widget, {widget_name=}"
                    )

        else:
            print(f"{task.result()=}")


def main():
    return BeeBeeware()


if __name__ == "__main__":
    main().main_loop()
