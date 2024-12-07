from typing import Literal
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import copy

class GeneticModuleMeta(ABC.__class__):
    """
    Metaclass to keep track of all GeneticModule subclasses.
    We can store them in a global registry to facilitate automatic discovery.
    """
    registry = {
        'leaf': [],
        'unary': [],
        'binary': [],
        'aggregate': []
    }

    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        # NOTE: could register here, instead currently use decorator with @register_module()
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

    def __init__(self, input_shapes, output_shape):
        super().__init__()
        # ensure these are simple tuples of ints
        input_shapes = tuple(tuple(int(s) for s in shape) for shape in input_shapes)
        output_shape = tuple(int(s) for s in output_shape)
        self.input_shapes = input_shapes
        self.output_shape = output_shape

    @abstractmethod
    def forward(self, *xs):
        pass

    @abstractmethod
    def mutate(self, module_factory):
        pass

    @staticmethod
    @abstractmethod
    def can_build(input_shapes, output_shape):
        pass

    @staticmethod
    @abstractmethod
    def infer_output_shape(input_shapes):
        pass

    @abstractmethod
    def complexity(self) -> float:
        pass

    def parameters_requires_grad(self, requires_grad):
        for p in self.parameters():
            p.requires_grad = requires_grad
        for sm in self.get_submodules():
            sm.parameters_requires_grad(requires_grad)

    def get_submodules(self):
        return []

    def set_submodule(self, index, new_module):
        pass

    def copy(self):
        # Use deepcopy to avoid recursion issues
        # PyTorch modules are usually deepcopy-safe
        return copy.deepcopy(self)

    def __str__(self):
        return self.__class__.__name__
