"""
An app for Bee.
"""
import asyncio
import contextlib
import copy
import importlib
import importlib.util
import inspect
import json
import pathlib
import sys
import time
import typing
from dataclasses import dataclass, field
from functools import partial, wraps
from io import StringIO
from types import ModuleType
from typing import Any, AsyncGenerator, Callable, Generator, OrderedDict, Union

import toga
import toga.handlers
import toga.paths
import toga.sources
import toga.validators
from huggingface_hub import hf_hub_download, list_models
from toga.colors import rgb
from toga.constants import Baseline
from toga.fonts import SANS_SERIF
from toga.style.pack import CENTER, COLUMN, ROW, Pack
from toga.widgets import textinput
from toga.widgets.table import OnSelectHandler

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


def timing(fun) -> Callable:
    @wraps(fun)
    def outer(*args, **kwargs):
        start = time.perf_counter()
        ret = fun(*args, **kwargs)
        end = time.perf_counter()
        duration = end - start
        print(f"---The execution of {fun.__name__=} {duration=}.---")
        return ret

    return outer


@timing
def update_config(
    instance: Union[toga.Widget, toga.Widget, None] = None,
    model_id: Union[str, None] = None,
    base_or_lora: str = "base",
) -> OrderedDict[str, Union[str, int, float, None, list, dict]]:
    AutoPipelineForText2Image = diffusers.AutoPipelineForText2Image
    pipe = AutoPipelineForText2Image

    # dir_dict_config = [entry for entry in dir(pipe) if "config" in entry]
    # print(dir_dict_config)

    if base_or_lora not in ["base", "lora"]:
        raise ValueError(f"Expected value from ['base', 'lora']. Got {base_or_lora=}")

    if instance:
        model_id = instance.config.get(f"{base_or_lora}_model")

    if model_id in ["", None]:
        raise ValueError(
            f"The model_id should be a valid id parameter from a model from huggingface_hub.list_models(). Got {model_id=}"
        )

    try:
        pipe_config = pipe.load_config(model_id, return_unused_kwargs=True)  # noqa: F841
    except OSError:
        pipe_config = {}  # noqa: F841
    # print(config)
    sig = inspect.signature(pipe.__call__)
    params = sig.parameters

    # params_dict = {param_name: (getattr(param_data.annotation, "get_args", None), param_data.default) for param_name, param_data in params.items()}

    params_dict = {}
    types = [list, int, str, float, dict]

    for param_name, param_data in params.items():
        if param_name in [
            "self",
            "kwargs",
            "callback_on_step_end_tensor_inputs",
            "ip_adapter_image",
            "latents",
            "generator",
        ]:
            continue

        default = param_data.default

        if typing.get_origin(param_data.annotation) is typing.Union:
            annotations = typing.get_args(param_data.annotation)
            annotations_out = [
                typing.get_origin(annotation) or annotation
                for annotation in annotations
            ]

        else:
            annotations = param_data.annotation
            annotations_out = [annotations]

        if not any(
            [
                (annotation in types) or (typing.get_origin(annotation) in types)
                for annotation in annotations_out
            ]
        ):
            continue

        params_dict[param_name] = (annotations_out, default)

    ret = OrderedDict(params_dict)

    if instance:
        instance.config.update(ret)

    return ret


@timing
def get_models(
    per_page: int,
    pipeline_tag: str,
    total_number: int,
    page_number: Union[int, None] = None,
    tags: Union[list[str], None] = None,
) -> Generator[Union[list[str], None], None, None]:
    filtr = copy.copy(tags) or []
    if default_pipeline not in filtr:
        filtr.append(default_pipeline)
    filtr.append(pipeline_tag)
    models = list_models(filter=filtr)
    models_list: list[str] = []
    models_counter: int = 0

    for index, model in enumerate(models):
        if page_number is None:
            page_number = yield page_number
            print(f"{page_number=}")

        if model.pipeline_tag != pipeline_tag:
            continue
        if page_number == total_number:
            # print(f"{page_number=}")
            break

        models_counter += 1

        if not isinstance(page_number, int):
            try:
                page_number = int(page_number)
            except TypeError as exception:
                print(exception)
                raise TypeError(
                    f"Expected page number of the type int. Got {type(page_number)=}."
                ) from exception

        if page_number > models_counter // per_page:
            continue

        models_list.append(model.id)
        # print(f"{model.id=}, {model.pipeline_tag=}")

        if (not int(len(models_list) % per_page)) and (len(models_list) > 0):
            ret = models_list.copy()
            models_list = []
            yield ret


@timing
def get_default_base_and_lora(
    pipeline_tag: Union[str, None] = None, tags: Union[list[str], None] = None
) -> tuple[str, str]:
    lora_filtr = copy.copy(tags) or []
    if "lora" not in lora_filtr:
        lora_filtr.append("lora")

    lora_filtr.append(str(pipeline_tag))

    if "StableDiffusionPipeline" not in lora_filtr:
        lora_filtr.append(default_pipeline)

    base_filtr = [str(pipeline_tag)]

    if "StableDiffusionPipeline" not in base_filtr:
        base_filtr.append(default_pipeline)

    base = [model.id for model in list_models(filter=base_filtr, limit=1)][0]
    lora = [model.id for model in list_models(filter=lora_filtr, limit=1)][0]
    return base, lora


@timing
def get_models_page(
    total_number: int = 100,
    page_num: int = 1,
    per_page: int = 5,
    pipeline_tag: str = "text-to-image",
    tags: Union[list[str], None] = None,
) -> Union[list[str], None]:
    page_num = int(page_num)
    models: Union[list[str], None] = []

    tags = tags or []

    if default_pipeline not in tags:
        tags.append(default_pipeline)

    if page_num <= 0:
        raise ValueError(f"Page number has to be greater than 0. Got {page_num=}")

    models_generator: Generator[Union[list[str], None], None, None] = get_models(
        per_page, pipeline_tag, total_number, tags=tags
    )

    models_generator.send(None)

    models = models or models_generator.send(page_num - 1)

    return models


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


no_preview_list: Callable[..., list[str]] = lambda: list(  # noqa: E731
    ["no_preview", "config_path", "base_model", "lora_model", "pipeline"]
)


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
    base_model: Union[str, None] = ""
    lora_model: Union[str, None] = ""
    pipeline: Union[str, None] = default_pipeline


main_config = Config()


class BeeBeeware(toga.App):
    placeholder_text = "Placeholder"
    main_window_split = {"menu": 1, "previews": 2}
    preview_container_split = {"menu": 1, "options": 1}
    config: OrderedDict[str, Union[str, None]] = OrderedDict(
        {
            config_name: config_value
            for config_name, config_value in main_config.__dict__.items()
        }
    )

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
        class check_path(textinput.OnConfirmHandler):
            def __call__(self, widget: toga.TextInput, **kwargs) -> toga.TextInput:
                path: pathlib.Path = pathlib.Path(widget.value)

                if not path.exists():
                    confirmation = toga.ConfirmDialog(
                        "Create a folder",
                        f"Do you wish to create the folder with the {path=}?",
                    )
                    if confirmation:
                        path.mkdir()
                return widget

        images_path: pathlib.Path = pathlib.Path(
            f"{toga.paths.Paths().config}\\training_data"
        )

        selected_path = toga.TextInput(  # noqa: F841
            id="source_path",
            style=Pack(direction=COLUMN),
            on_confirm=check_path(),
            readonly=True,
        )

        if not images_path.exists():
            images_path.mkdir()

        select_path = toga.Button(
            "Select path", id="source_path_button", on_press=self.path_handler
        )

        selection_box_paths = toga.Box(
            style=Pack(direction=ROW),
            children=[selected_path, select_path],
        )

        self.previews_container.content = selection_box_paths

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

        model_configs: OrderedDict[
            str, Union[str, int, float, list, dict, None]
        ] = update_config(instance=self, model_id=base_model)
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
            validators: list[
                Union[toga.validators.CountValidator, toga.validators.BooleanValidator]
            ] | None = []
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
        default: dict[str, str] = {
            "Save": str(toga.paths.Paths().config),
            "Load": str(toga.paths.Paths().config),
        }
        text_window = toga.Window(title=window_name)

        select_path = toga.Button(
            f"{window_name} path",
            id=f"{window_name}_path_button",
            on_press=self.path_handler,
        )
        selected_path = toga.TextInput(id=f"{window_name}_path", readonly=True)

        entry_box = toga.Box(id=window_name, style=Pack(direction=ROW))
        text_input_box = toga.TextInput(placeholder=f"{window_name}")
        path: str = default[window_name]

        confirm_button = toga.Button(
            "Confirm",
            on_press=partial(
                self.close_window,
                window=text_window,
                path=path,
                text_input_box=text_input_box,
            ),
        )

        entry_box.add(text_input_box)
        entry_box.add(confirm_button)

        path_box = toga.Box(
            style=Pack(direction=ROW), children=[selected_path, select_path]
        )

        text_window.content = toga.Box(
            style=Pack(direction=COLUMN), children=[entry_box, path_box]
        )

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

        config_name = f"{text_input_box.value}.json" or "beeconfig.json"

        save_load_path = self.main_window.widgets[f"{window.title}_path"]

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

    def path_handler(self, widget, **kwargs):
        images_path: pathlib.Path = pathlib.Path(
            f"{toga.paths.Paths().config}\\training_data"
        )
        source_path = toga.SelectFolderDialog(
            "Select training data folder", initial_directory=images_path
        )

        task_name = str(widget.id).replace("_button", "")

        task = asyncio.create_task(self.main_window.dialog(source_path), name=task_name)
        task.add_done_callback(self.dialog_dismissed)
        print("Dialog has been created")

    def dialog_dismissed(self, task):
        widget_name: str = task.get_name()

        if task.result():
            print(f"{task.result()=}")
            self.main_window.widgets[widget_name].value = task.result()
        else:
            print(f"{task.result()=}")


def main():
    return BeeBeeware()


if __name__ == "__main__":
    main().main_loop()
