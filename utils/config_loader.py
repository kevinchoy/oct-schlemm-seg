import importlib
from typing import Any
from omegaconf import DictConfig

def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """Extract an object from a given path.
        https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(
            "Object `{}` cannot be loaded from `{}`.".format(obj_name, obj_path)
        )
    return getattr(module_obj, obj_name)


def load_dataset(cfg: DictConfig, transform=None) -> object:
    obj = load_obj(cfg.name)
    # TODO: config for transforms
    if not cfg.transform:
        # cfg.params.transform = transform
        transform = None
    if 'params' in cfg.keys() and cfg.params is not None:
        dataset = obj(**cfg.params, transform=transform)
    else:
        dataset = obj()
    return dataset


def load_model(cfg: DictConfig) -> object:
    obj = load_obj(cfg.name)
    if 'params' in cfg.keys() and cfg.params is not None:
        model = obj(**cfg.params)
    else:
        model = obj()
    return model


def load_loss(cfg: DictConfig) -> object:
    obj = load_obj(cfg.name)
    if 'params' in cfg.keys() and cfg.params is not None:
        loss = obj(**cfg.params)
    else:
        loss = obj()
    return loss


def load_optimizer(model, cfg: DictConfig) -> object:
    """

    :param model:
    :param cfg: cfg = cfg.optimizer, contains keys: name, params
    :return:
    """
    obj = load_obj(cfg.name)
    if 'params' in cfg.keys() and cfg.params is not None:
        optimizer = obj(model.parameters(), **cfg.params)
    else:
        optimizer = obj(model.parameters())
    return optimizer
