from typing import Literal, Optional, Self
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import copy

type shape = tuple[int, ...]

class GeneticModuleMeta(ABC.__class__):
    """
    Metaclass to keep track of all GeneticModule subclasses.
    We can store them in a global registry to facilitate automatic discovery.
    """
    global_registry = []

    registry = {
        'leaf': [],
        'unary': [],
        'binary': [],
        'aggregate': []
    }

    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        mcs.global_registry.append(cls)
        return cls

def register_module(category: Literal['leaf', 'unary', 'binary', 'aggregate']):
    """
    Decorator to register a module class in a given category.
    category should be one of: 'leaf', 'unary', 'binary', 'aggregate'.
    """
    def decorator(cls):
        if hasattr(cls, 'can_build'):
            GeneticModuleMeta.registry[category].append(cls)
        return cls
    return decorator

class GeneticModule(ABC, nn.Module, metaclass=GeneticModuleMeta):
    """
    Base class for genetic modules. 
    """

    N_CHILDREN: int

    def __init__(self, input_shapes: shape, output_shape: shape, children: Optional[list[Self]] = None):
        super().__init__()
        self.input_shapes = input_shapes
        self.output_shape = output_shape
        self.submodules = children if children else []

    @abstractmethod
    def forward(self, *xs): ...

    @abstractmethod
    def mutate(self, module_factory): ...

    @staticmethod
    @abstractmethod
    def can_build(input_shapes, output_shape) -> bool: ...

    @staticmethod
    @abstractmethod
    def infer_output_shape(input_shapes) -> shape: ...

    @abstractmethod
    def complexity(self) -> float: ...

    def compute_complexity(self):
        return self.complexity() + sum(sm.compute_complexity() for sm in self.submodules)

    def parameters_requires_grad(self, requires_grad):
        for p in self.parameters():
            p.requires_grad = requires_grad
        for sm in self.submodules:
            sm.parameters_requires_grad(requires_grad)

    def copy(self):
        # Use deepcopy to avoid recursion issues
        # PyTorch modules are usually deepcopy-safe
        return copy.deepcopy(self)

    def __str__(self):
        return self.__class__.__name__
