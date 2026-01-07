import os
import time
from functools import wraps
from types import ModuleType
from typing import Generator  # noqa
from typing import OrderedDict  # noqa
from typing import Any, AsyncGenerator, Callable, TypeAlias, Union  # noqa

import clr
import toga

from ..helpers.managers import capture

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


def capture_decorator(fun) -> Callable:
    @wraps(fun)
    def outer(*args, **kwargs):
        widget_out = args[0]
        widget = args[1]
        if not isinstance(widget_out, toga.Widget):
            raise TypeError(f"Expected a toga.Widget instance. Got {widget_out=}.")

        def inner(*args, **kwargs):
            return fun(args, kwargs)

        if not isinstance(widget, toga.Widget):
            return inner(args, kwargs)

        def inner_captured(*args, **kwargs):
            print(f"inner_captured(): {args=}.")
            with capture() as out:  # noqa: F841
                widget, *_ = args
                args = tuple(args[1:])
                print(f"capture_decorator(): {widget.id=}.")
                ret = fun(*args, **kwargs)

            for prnt in out:
                widget.value = prnt
                print(f"Captured output: {widget.value, prnt=}.")
                widget.refresh()

            return ret

        return inner_captured(*args, **kwargs)

    return outer
