"""
An app for Bee.
"""

import asyncio
import toga
import toga.paths
import pathlib
import json
import copy

from decimal import Decimal
from dataclasses import dataclass
from typing import Coroutine
from functools import partial, wraps

from toga.style.pack import CENTER, COLUMN, ROW, Pack
from toga.colors import WHITE, rgb
from toga.constants import Baseline
from toga.fonts import SANS_SERIF
from toga.style import Pack

from typing import OrderedDict, Union, Optional, Generator
from huggingface_hub import list_models


def get_models(per_page: int, pipeline_tag: str, total_number: int, page_number: Union[int, None] = None) -> Generator[Union[list[str], None], None, None]:

    models = list_models()
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
                raise TypeError(f"Expected page number of the type int. Got {type(page_number)=}.") from exception

        if page_number > models_counter // per_page:

            continue

        models_list.append(model.id)
        # print(f"{model.id=}, {model.pipeline_tag=}")

        if (not int(len(models_list) % per_page)) and (len(models_list) > 0):
            ret = models_list.copy()
            models_list = []
            yield ret


def get_models_page(total_number: int, page_num: int, per_page: int, pipeline_tag: str = "text-to-image") -> Union[list[str], None]:

    page_num = int(page_num)
    models: Union[list[str], None] = []

    if page_num <= 0:
        raise ValueError(f"Page number has to be greater than 0. Got {page_num=}")

    models_generator: Generator[Union[list[str], None], None, None] = get_models(per_page, pipeline_tag, total_number)

    models_generator.send(None)

    models = models or models_generator.send(page_num - 1)

    return models


def assign_container(fun):

    @wraps(fun)
    def outer(container=None):

        def inner(instance, widget, *args, **kwargs):
            return fun(instance, widget, *args, container=container, **kwargs)
        return inner

    return outer


@dataclass
class Config:
    config_path: Union[str, pathlib.Path] = ""
    placholder_option_1: str = ""
    placeholder_option_2: str = ""


class BeeBeeware(toga.App):

    placeholder_text = "Placeholder"
    main_window_split = {"menu": 1, "previews": 2}
    preview_container_split = {"menu": 1, "options": 1}
    text_input_box: toga.TextInput

    def startup(self):
        """Construct and show the Toga application.

        Usually, you would add your application to a main content box.
        We then create a main window (with a name matching the app), and
        show the main window.
        """

        self.main_window = toga.MainWindow(title=self.formal_name)

        self.main_buttons = OrderedDict({
            "Model": self.preview_model_menu,
            "Config": self.preview_config,
            })
        
        self.aux_buttons = OrderedDict({
            "Load from file": self.load_config,
            "Save to file": self.save_config,
            "Next": self.next,
            "Previous": self.previous,
            })

        main_box = toga.Box(style=Pack(direction=COLUMN))

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

        previews = toga.Box(style=Pack(direction=COLUMN, alignment=CENTER))
        self.previews_container = toga.ScrollContainer(id="previews_container", horizontal=False, style=Pack(direction=COLUMN))
        self.previews_container.content = previews

        menu_previews_split.content = [(menu, self.main_window_split["menu"]), (self.previews_container, self.main_window_split["previews"])]

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
        base_models_data = [f"Base palceholder {index}" for index in range(2)]
        lora_models_data = [f"lora placeholder {index}" for index in range(4)]

        base_models_table = toga.Table(id="base_models", headings=["Base models"], data=base_models_data)
        base_models_next = toga.Button(text="Next", on_press=self.aux_buttons["Next"](container=base_models_table))
        base_models_prev = toga.Button(text="Previous", on_press=self.aux_buttons["Previous"](container=base_models_table))
        base_buttons = toga.Box(id="base_buttons", style=Pack(direction=ROW), children=[base_models_prev, base_models_next])
        base_models_box = toga.Box(id="base_box", style=Pack(direction=COLUMN), children=[base_models_table, base_buttons])

        lora_models_table = toga.Table(id="lora_models", headings=["Trainable model"], data=lora_models_data)
        lora_models_next = toga.Button(text="Next", on_press=self.aux_buttons["Next"](container=lora_models_table))
        lora_models_prev = toga.Button(text="Previous", on_press=self.aux_buttons["Previous"](container=lora_models_table))
        lora_buttons = toga.Box(id="lora_buttons", style=Pack(direction=ROW), children=[lora_models_prev, lora_models_next])
        lora_models_box = toga.Box(id="lora_box", style=Pack(direction=COLUMN), children=[lora_models_table, lora_buttons])

        models = toga.SplitContainer(id="models", style=Pack(direction=COLUMN))
        models.content = [(base_models_table, self.preview_container_split["menu"]), (lora_models_table, self.preview_container_split["options"])]
        self.previews_container.content = models

    def preview_config(self, widget):

        config = OrderedDict({
            "Placeholder 1": toga.NumberInput(min=0, max=10, step=0.1, value=1),
            "Placeholder 2": toga.NumberInput(min=0, max=10, step=0.1, value=1)
        })

        config_scroll = toga.Box(id="config", style=Pack(direction=COLUMN))
        save_button = toga.Button("Save to file", on_press=self.aux_buttons["Save to file"])
        load_button = toga.Button("Load from file", on_press=self.aux_buttons["Load from file"])

        for config_name, config_input in config.items():
            label = toga.Label(config_name)
            config_box = toga.Box(style=Pack(direction=ROW))

            config_box.add(label)
            config_box.add(config_input)

            config_scroll.add(config_box)

        save_load_box = toga.Box(id="save_and_load", style=Pack(direction=ROW))
        save_load_box.add(save_button)
        save_load_box.add(load_button)

        config_scroll.add(save_load_box)

        self.previews_container.content = config_scroll

    @assign_container
    def next(self, widget, container: Union[toga.Box, toga.Table, None] = None) -> Union[toga.Box, toga.Table]:
        print(f"{container.id=}, {container.id in self.main_window.content.children=}")
        raise NotImplementedError

    @assign_container
    def previous(self, widget, container: Union[toga.Box, toga.Table, None] = None) -> Union[toga.Box, toga.Table]:
        print(f"{container.id=}, {container.id in self.main_window.content.children=}")
        raise NotImplementedError

    def text_input(self, widget, window_name: str = "") -> toga.Window:
        default: dict[str, str] = {"Save": str(toga.paths.Paths().config), "Load": str(toga.paths.Paths().config)}
        text_window = toga.Window(title=window_name)
        
        entry_box = toga.Box(id=window_name, style=Pack(direction=ROW))
        text_input_box = toga.TextInput(placeholder=f"{window_name}")
        path: str = default[window_name]

        confirm_button = toga.Button("Confirm", on_press=partial(self.close_window, window=text_window, path=path, text_input_box=text_input_box))

        entry_box.add(text_input_box)
        entry_box.add(confirm_button)

        text_window.content = entry_box

        text_window.show()

        return text_window
    
    async def load_config(self, widget) -> None:

        self.text_input(widget, "Load")

    async def save_config(self, widget) -> None:

        self.text_input(widget, window_name="Save")

    def close_window(self, widget, window: toga.Window, path: str, text_input_box: toga.TextInput, config: Config = Config()) -> Config:
        print(f"Closing {window.title=}")

        path = text_input_box.value or path
        config_path: Union[str, pathlib.Path] = ""
        if window.title == "Load":
            
            if (config.config_path == "") or (path is not None):
                config.config_path = path

            config_path = config.config_path

            config_path = pathlib.Path(config_path)

            if not config_path.exists() and (config_path is not None):
                print(f"Creating {config_path=}")
                pathlib.Path(f"{config_path}\\").mkdir()
                placeholder = {"configs": None}
                json_configs = json.dumps(placeholder)

                with open(pathlib.Path(f"{config_path}\\beeconfig.json"), "+w") as config_file:
                    config_file.write(json_configs)

            if config_path is not None:
                config_dict: OrderedDict = OrderedDict({})
                with open(pathlib.Path(f"{config_path}\\beeconfig.json"), 'r') as config_file:
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
                placeholder = {"configs": None}
                json_configs = json.dumps(placeholder)

                with open(pathlib.Path(f"{config_path}\\beeconfig.json"), "+w") as config_file:
                    config_file.write(json_configs)

            if config_path is not None:
                config_to_save = copy.copy(config.__dict__)
                config_to_save["config_path"] = str(config.__dict__["config_path"])
                json_config = json.dumps(config_to_save)

                with open(pathlib.Path(f"{config_path}\\beeconfig.json"), "+w") as config_file:
                    config_file.write(json_config)

                print(f"Config file saved at {config_path=}")
        
        window.close()
        return config
    
    def draw_text(self, widget):
        print("Writing on canvas.")
        self.previews_container.content = self.canvas
        font = toga.Font(family=SANS_SERIF, size=20)
        self.text_width, text_height = self.canvas.measure_text(self.placeholder_text, font)

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
            self.text = text_filler.write_text(self.placeholder_text, x, y, font, Baseline.TOP)


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


def main():
    return BeeBeeware()


if __name__ == "__main__":
    main().main_loop()

