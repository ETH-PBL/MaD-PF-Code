from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Type

import torch
from torch import nn
from typing_extensions import TypeVar

T_Config = TypeVar('T_Config', bound='ModuleConfig')
# Unbounded: register also decorates ResampleConfig (not a ModuleConfig); an
# identity signature preserves each decorated class's concrete type.
T = TypeVar('T')


class ConfigRegistry:
  _registry: dict[str, type] = {}

  @classmethod
  def register(cls, subclass: type[T]) -> type[T]:
    key = subclass.__name__
    if key in cls._registry:
      raise ValueError(f'duplicate config registration for: {key}')
    cls._registry[key] = subclass
    return subclass

  @classmethod
  def get(cls, name: str) -> type:
    if name not in cls._registry:
      raise KeyError(f'configuration {name} is not registered')
    return cls._registry[name]

  @classmethod
  def is_config(cls, value) -> bool:
    """True if `value` is an instance of a registered config class."""
    return type(value) in cls._registry.values()

  @classmethod
  def encode(cls, value):
    """Recursively serialize to a plain structure, tagging every registered
    config (ModuleConfig or ResampleConfig) with its class name under '_cls'.

    Lets polymorphic and nested configs round-trip without relying on field
    type annotations (the abstract base of a polymorphic field carries no
    concrete type, and asdict would drop it).
    """
    if cls.is_config(value):
      return {'_cls': type(value).__name__,
              **{f.name: cls.encode(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, (list, tuple)):
      return type(value)(cls.encode(v) for v in value)
    if isinstance(value, dict):
      return {k: cls.encode(v) for k, v in value.items()}
    return value

  @classmethod
  def decode(cls, data):
    """Inverse of `encode`: rebuild any '_cls'-tagged config via the registry."""
    if isinstance(data, dict) and '_cls' in data:
      config_class = cls.get(data['_cls'])
      return config_class(**{k: cls.decode(v) for k, v in data.items() if k != '_cls'})
    if isinstance(data, (list, tuple)):
      return type(data)(cls.decode(v) for v in data)
    if isinstance(data, dict):
      return {k: cls.decode(v) for k, v in data.items()}
    return data


@dataclass
class ModuleConfig(ABC):

  @abstractmethod
  def build_model(self) -> nn.Module:
    """
    produce a PyTorch model instance according to this configuration's parameters
    """

  def save_module(self, model: nn.Module, path: Path):
    """
    save this config and the given model weights to a PyTorch file

    @param model: PyTorch model instance (should originate from `build_model()`)
    @param path: file path to save to
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
      'state_dict': model.state_dict(),
      'config': ConfigRegistry.encode(self),
    }, path)

  @classmethod
  def load_config(cls: Type[T_Config], path: str | Path) -> T_Config:
    """
    load config from a saved file

    @param path: file path to load from
    """
    data = torch.load(path, map_location='cpu')

    if 'config' not in data:
      raise ValueError(f'checkpoint at {path} does not contain config metadata')

    config = ConfigRegistry.decode(data['config'])

    if not isinstance(config, cls) and cls is not ModuleConfig:
      raise TypeError(f'expected config type {cls.__name__}, loaded {config.__class__.__name__}')

    return config

  @staticmethod
  def apply_state_dict(model: nn.Module, path: Path):
    """
    load weights and apply them to a PyTorch model

    @param model: target PyTorch model
    @param path: file path to load from
    """
    data = torch.load(path, map_location='cpu')
    model.load_state_dict(data['state_dict'], strict=False)

  @classmethod
  def load_module(cls: Type[T_Config], path: Path) -> tuple[T_Config, nn.Module]:
    """
    load both the configuration and the model weights from a file

    @param path: file path to load from
    @return: instance of the configuration and a model with applied weights
    """
    config = cls.load_config(path)
    module = config.build_model()
    cls.apply_state_dict(module, path)
    return config, module
