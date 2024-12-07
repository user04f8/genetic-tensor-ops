import torch
import torch.nn as nn
import math
import random
from genetic_module import GeneticModule, register_module
from shape_utils import shapes_equal, can_elementwise, can_select_input, can_reduce

@register_module('aggregate')
class SumGM(GeneticModule):
    def __init__(self, input_shapes, output_shape, child_f, dim=None):
        super().__init__(input_shapes, output_shape)
        self.child_f = child_f
        self.dim = dim

    def forward(self, *xs):
        out = self.child_f(*xs)
        return out.sum(dim=self.dim)

    def get_submodules(self):
        return [self.child_f]

    def set_submodule(self, index, new_module):
        self.child_f = new_module

    def mutate(self, module_factory):
        if random.random() < 0.5:
            self.child_f = self.child_f.mutate(module_factory)
        return self

    @staticmethod
    def can_build(input_shapes, output_shape):
        # need to ensure that there's a dimension sum can remove
        # Simplified: just check can_reduce on first input shape
        return any(can_reduce(s, output_shape) for s in input_shapes)

    @staticmethod
    def infer_output_shape(input_shapes):
        # Without chosen dim, can't guarantee. Factory chooses dim after deciding
        return None

    def complexity(self):
        return 4.0 + self.child_f.complexity()

    def __str__(self):
        return f'sum({self.child_f}, dim={self.dim})'

# TODO: AvgGM, NormGM would follow same pattern
@register_module('aggregate')
class MaxGM(GeneticModule):
    def __init__(self, input_shapes, output_shape, child_f, dim=None):
        super().__init__(input_shapes, output_shape)
        self.child_f = child_f
        self.dim = dim

    def forward(self, *xs):
        out = self.child_f(*xs)
        if self.dim is None:
            # Global max over all elements
            # Just out.max() returns a scalar
            return out.max()
        else:
            # If dim is specified, out.max(dim) returns (values, indices)
            return out.max(dim=self.dim).values
    
    def get_submodules(self):
        return [self.child_f]

    def set_submodule(self, index, new_module):
        self.child_f = new_module

    def mutate(self, module_factory):
        if random.random() < 0.5:
            self.child_f = self.child_f.mutate(module_factory)
        return self

    @staticmethod
    def can_build(input_shapes, output_shape):
        return any(can_reduce(s, output_shape) for s in input_shapes)

    @staticmethod
    def infer_output_shape(input_shapes):
        return None

    def complexity(self):
        return 4.0 + self.child_f.complexity()

    def __str__(self):
        return f'max({self.child_f}, dim={self.dim})'
