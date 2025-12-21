import copy
import inspect
import os
import typing
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
from huggingface_hub import list_models

from .decorators import timing

# import diffusers


clr.AddReference("System.Drawing")  # noqa
from System.Drawing import Image as WinImage  # noqa

PathLikeT: TypeAlias = str | os.PathLike
BytesLikeT: TypeAlias = bytes | bytearray | memoryview
ImageLikeT: TypeAlias = Any
ImageContentT: TypeAlias = PathLikeT | BytesLikeT | ImageLikeT

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
