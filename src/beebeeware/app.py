"""
An app for Bee.
"""

import asyncio
import toga
from decimal import Decimal


from toga.style.pack import CENTER, COLUMN, ROW, Pack
from toga.colors import WHITE, rgb
from toga.constants import Baseline
from toga.fonts import SANS_SERIF
from toga.style import Pack

from typing import OrderedDict




class BeeBeeware(toga.App):

    placeholder_text = "Placeholder"
    main_window_split = {"menu": 1, "previews": 2}
    preview_container_split = {"menu": 1, "options": 1}

    def startup(self):
        """Construct and show the Toga application.

        Usually, you would add your application to a main content box.
        We then create a main window (with a name matching the app), and
        show the main window.
        """

        self.main_window = toga.MainWindow(title=self.formal_name)

        self.buttons = OrderedDict({
            "Model": self.preview_model_menu,
            "Config": self.preview_config})
        main_box = toga.Box(style=Pack(direction=COLUMN))

        menu_previews_split = toga.SplitContainer(style=Pack(direction=COLUMN))

        menu = toga.Box(id="menu", style=Pack(direction=COLUMN, alignment=CENTER))

        for button_name, button_action in self.buttons.items():
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

        base_models_table = toga.Table(headings=["Base models"], data=base_models_data)
        lora_models_table = toga.Table(headings=["Trainable model"], data=lora_models_data)
        models = toga.SplitContainer(id="models", style=Pack(direction=COLUMN))
        models.content = [(base_models_table, self.preview_container_split["menu"]), (lora_models_table, self.preview_container_split["options"])]
        self.previews_container.content = models

    def preview_config(self, widget):

        config = OrderedDict({
            "Placeholder 1": toga.NumberInput(min=0, max=10, step=0.1, value=1),
            "Placeholder 2": toga.NumberInput(min=0, max=10, step=0.1, value=1)
        })

        config_scroll = toga.Box(id="config", style=Pack(direction=COLUMN))

        for config_name, config_input in config.items():
            label = toga.Label(config_name)
            config_box = toga.Box(style=Pack(direction=ROW))

            config_box.add(label)
            config_box.add(config_input)

            config_scroll.add(config_box)

        self.previews_container.content = config_scroll

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
