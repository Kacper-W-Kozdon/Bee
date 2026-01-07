import asyncio
import importlib
import pathlib
import sys
from os import listdir
from os.path import isfile, join
from typing import Generator  # noqa
from typing import OrderedDict  # noqa
from typing import Any, AsyncGenerator, Callable, TypeAlias, Union  # noqa

import toga

from ..helpers.decorators import capture_decorator


def list_files(
    mypath: Union[str, pathlib.Path], format_: str = "str", extension: str = ".json"
) -> Union[list[str], list[pathlib.Path], list]:
    # Source - https://stackoverflow.com/a
    # Posted by pycruft, modified by community. See post 'Timeline' for change history
    # Retrieved 2025-11-24, License - CC BY-SA 4.0

    valid_formats_ = ["Path", "str", "path", "string", "file"]
    valid_extensions = [".jpg", ".json", ".png", ".jpeg", ""]

    print(
        f"Listing {extension=} files in the location {mypath=}. Formatting {format_=}."
    )

    if format_ not in valid_formats_:
        raise ValueError(
            f"Incorrect value for the format_ parameter. Expected one of the {valid_formats_=}. Got {format_=}."
        )

    if extension not in valid_extensions:
        raise ValueError(
            f"Incorrect value for the extension parameter. Expected one of the {valid_extensions=}. Got {extension=}."
        )

    onlyfiles: Union[list[str], list[pathlib.Path], list] = []

    if not mypath:
        return onlyfiles

    match format_:
        case "path" | "Path":
            onlyfiles = [
                pathlib.Path(f"{str(mypath)}\\{f}")
                for f in listdir(mypath)
                if (isfile(join(mypath, f)) and extension in f)
            ]

        case "str" | "string":
            onlyfiles = [
                f"{str(mypath)}\\{f}"
                for f in listdir(mypath)
                if (isfile(join(mypath, f)) and extension in f)
            ]

        case "file":
            onlyfiles = [
                f
                for f in listdir(mypath)
                if (isfile(join(mypath, f)) and extension in f)
            ]

    return onlyfiles


# pylint: disable-next=too-few-public-methods
class Loader:
    """
    Docstring for Loader: delays loading the libs without triggering
    pylint, mypy or pylance.
    """

    def __init__(self, loadable: str):
        self.loadable = loadable

    def __await__(self):
        lib = self.loadable
        print(f"Loading {lib}")
        if lib not in sys.modules:
            globals().update({lib: importlib.import_module(lib)})

        return (yield None)


async def loader(libraries: list[str], counter: int = 0):
    """
    Docstring for loader: delays loading the libs without triggering
    pylint, mypy or pylance.

    :param libraries: Description
    :type libraries: list[str]
    :param counter: Description
    :type counter: int
    """
    for lib in libraries:
        print(f"{lib=}, {lib in sys.modules=}")

        if lib not in sys.modules:
            await Loader(lib)
        counter += 1

        print(f"{lib in sys.modules=}")
        yield counter


@capture_decorator
async def load_libs(widget: toga.Widget, libraries: list[str]):
    """
    Docstring for load_libs: delays loading the libs without triggering
    pylint, mypy or pylance.

    :param libraries: Description
    :type libraries: list[str]
    :param widget: Description
    :type widget: toga.Widget
    """
    async for item in loader(libraries):
        widget.value = item

        print(widget.value)

        await asyncio.sleep(0.1)
