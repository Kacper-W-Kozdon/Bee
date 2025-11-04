"""
An app for Bee.
"""

import asyncio
import toga

from toga.style.pack import CENTER, COLUMN, ROW, Pack
from toga.colors import WHITE, rgb
from toga.constants import Baseline
from toga.fonts import SANS_SERIF
from toga.style import Pack

from typing import OrderedDict




class BeeBeeware(toga.App):

    placeholder_text = "Placeholder"

    def startup(self):
        """Construct and show the Toga application.

        Usually, you would add your application to a main content box.
        We then create a main window (with a name matching the app), and
        show the main window.
        """
        self.buttons = OrderedDict({"Placeholder button": self.draw_text})
        main_box = toga.Box()

        menu_previews_split = toga.SplitContainer()
        menu = toga.Box(style=Pack(direction=COLUMN, margin_top=50))

        for button_name, button_action in self.buttons.items():
            menu.add(
                toga.Button(
                    f"{button_name}",
                    on_press=button_action,
                    style=Pack(width=200, margin=20),
                )
            )

        self.previews = toga.Box(style=Pack(direction=COLUMN, margin_top=50))

        menu_previews_split.content = [(menu, 1), (self.previews, 1)]

        main_box.add(menu_previews_split)

        self.canvas = toga.Canvas(
            style=Pack(flex=1),
            on_resize=self.on_resize,
            on_press=self.on_press,
        )

        self.main_window = toga.MainWindow(title=self.formal_name)
        self.main_window.content = main_box
        self.main_window.show()

    def draw_text(self, widget):
        self.previews.content = self.canvas
        font = toga.Font(family=SANS_SERIF, size=20)
        self.text_width, text_height = self.canvas.measure_text(self.placeholder_text, font)

        x = (150 - self.text_width) // 2
        y = 175

        with self.canvas.Stroke(color="REBECCAPURPLE", line_width=4.0) as rect_stroker:
            self.text_border = rect_stroker.rect(
                x - 5,
                y - 5,
                self.text_width + 10,
                text_height + 10,
            )
        with self.canvas.Fill(color=rgb(149, 119, 73)) as text_filler:
            self.text = text_filler.write_text(self.placeholder_text, x, y, font, Baseline.TOP)

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
