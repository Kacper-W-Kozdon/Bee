import pathlib
from os import listdir
from os.path import isfile, join
from typing import Generator  # noqa
from typing import OrderedDict  # noqa
from typing import Any, AsyncGenerator, Callable, TypeAlias, Union  # noqa


def list_files(
    mypath: Union[str, pathlib.Path], format_: str = "Path", extension: str = ".json"
) -> Union[list[str], list[pathlib.Path], list]:
    # Source - https://stackoverflow.com/a
    # Posted by pycruft, modified by community. See post 'Timeline' for change history
    # Retrieved 2025-11-24, License - CC BY-SA 4.0

    valid_formats_ = ["Path", "str", "path", "string", "file"]
    valid_extensions = [".jpg", ".json", ".png", ".jpeg", ""]

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
