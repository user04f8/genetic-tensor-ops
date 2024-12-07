import torch
import torch.nn as nn
import math
import random
from gm.genetic_module import GeneticModule, register_module
from shape_utils import shapes_equal, can_elementwise, can_select_input, can_reduce

class BinaryElementwiseGM(GeneticModule):
    # A base class for elementwise binary ops (Add, Mult)
    # so we don't have to re-implement complexity, mutation strategies
    def __init__(self, input_shapes, output_shape, child_f, child_g):
        super().__init__(input_shapes, output_shape, [child_f, child_g])
        self.child_f = child_f
        self.child_g = child_g

    def mutate(self, module_factory):
        if random.random() < 0.5:
            self.child_f = self.child_f.mutate(module_factory)
        if random.random() < 0.5:
            self.child_g = self.child_g.mutate(module_factory)
        if random.random() < 0.05:
            candidate = module_factory.create_random_module(self.child_f.input_shapes, self.child_f.output_shape)
            if candidate:
                self.child_f = candidate
        if random.random() < 0.05:
            candidate = module_factory.create_random_module(self.child_g.input_shapes, self.child_g.output_shape)
            if candidate:
                self.child_g = candidate
        if random.random() < 0.01:
            candidate = module_factory.create_random_module(self.input_shapes, self.output_shape)
            if candidate:
                return candidate
        return self

    @staticmethod
    def can_build(input_shapes, output_shape):
        return can_elementwise(input_shapes, output_shape)

    @staticmethod
    def infer_output_shape(input_shapes):
        # If all equal
        s = input_shapes[0]
        if all(s == ish for ish in input_shapes):
            return s
        return None

    def complexity(self):
        # base complexity + children complexity
        return 5.0

@register_module('binary')
class MultGM(BinaryElementwiseGM):
    def forward(self, *xs):
        return self.child_f(*xs) * self.child_g(*xs)

    def __str__(self):
        return f'{self.child_f}*{self.child_g}'

@register_module('binary')
class AddGM(BinaryElementwiseGM):
    def forward(self, *xs):
        return self.child_f(*xs) + self.child_g(*xs)

    def __str__(self):
        return f'({self.child_f}+{self.child_g})'

@register_module('unary')
class NegateGM(GeneticModule):
    # Unary op: output = -child_f
    def __init__(self, input_shapes, output_shape, child_f):
        super().__init__(input_shapes, output_shape)
        self.child_f = child_f

    def forward(self, *xs):
        return -self.child_f(*xs)

    def get_submodules(self):
        return [self.child_f]

    def set_submodule(self, index, new_module):
        if index == 0:
            self.child_f = new_module

    def mutate(self, module_factory):
        if random.random() < 0.05:
            return self.child_f  # remove negate
        if random.random() < 0.5:
            self.child_f = self.child_f.mutate(module_factory)
        return self

    @staticmethod
    def can_build(input_shapes, output_shape):
        # child must produce output_shape
        return True

    @staticmethod
    def infer_output_shape(input_shapes):
        # same shape as child
        return None

    def complexity(self):
        return 3.0 + self.child_f.complexity()

    def __str__(self):
        return f'-({self.child_f})'

@register_module('unary')
class SineGM(GeneticModule):
    def __init__(self, input_shapes, output_shape, child_f):
        super().__init__(input_shapes, output_shape)
        self.child_f = child_f

    def forward(self, *xs):
        return torch.sin(self.child_f(*xs))

    def get_submodules(self):
        return [self.child_f]

    def set_submodule(self, index, new_module):
        self.child_f = new_module

    def mutate(self, module_factory):
        if random.random() < 0.05:
            return self.child_f  # remove sine
        if random.random() < 0.5:
            self.child_f = self.child_f.mutate(module_factory)
        return self

    @staticmethod
    def can_build(input_shapes, output_shape):
        return True

    @staticmethod
    def infer_output_shape(input_shapes):
        return None

    def complexity(self):
        return 5.0 + self.child_f.complexity()

    def __str__(self):
        return f'sin({self.child_f})'
    

@register_module('unary')
class IncrementGM(GeneticModule):
    # Unary op: output = -child_f
    def __init__(self, input_shapes, output_shape, child_f):
        super().__init__(input_shapes, output_shape, [child_f])
        self.child_f = child_f

    def forward(self, *xs):
        return self.child_f(*xs) + 1

    def set_submodule(self, index, new_module):
        if index == 0:
            self.child_f = new_module

    def mutate(self, module_factory):
        if random.random() < 0.5:
            self.child_f = self.child_f.mutate(module_factory)
        return self

    @staticmethod
    def can_build(input_shapes, output_shape):
        # child must produce output_shape
        return True

    @staticmethod
    def infer_output_shape(input_shapes):
        # same shape as child
        return None

    def complexity(self):
        return 1.0

    def __str__(self):
        return f'({self.child_f}+1)'

@register_module('binary')
class GreaterThanGM(GeneticModule):
    # Outputs a boolean mask: child_f(*xs) > child_g(*xs)
    # For simplicity, output is float {0,1}.
    def __init__(self, input_shapes, output_shape, child_f, child_g):
        super().__init__(input_shapes, output_shape, [child_f, child_g])
        self.child_f = child_f
        self.child_g = child_g

    def forward(self, *xs):
        return (self.child_f(*xs) > self.child_g(*xs)).float()

    def get_submodules(self):
        return [self.child_f, self.child_g]

    def set_submodule(self, index, new_module):
        if index == 0:
            self.child_f = new_module
        else:
            self.child_g = new_module

    def mutate(self, module_factory):
        if random.random() < 0.5:
            self.child_f = self.child_f.mutate(module_factory)
        if random.random() < 0.5:
            self.child_g = self.child_g.mutate(module_factory)
        return self

    @staticmethod
    def can_build(input_shapes, output_shape):
        return can_elementwise(input_shapes, output_shape)

    @staticmethod
    def infer_output_shape(input_shapes):
        return BinaryElementwiseGM.infer_output_shape(input_shapes)

    def complexity(self):
        return 5.0

    def __str__(self):
        return f'({self.child_f}>{self.child_g})'

@register_module('binary')
class DotProductGM(GeneticModule):
    # Compute dot product along last dimension if shapes match
    def __init__(self, input_shapes, output_shape, child_f, child_g):
        super().__init__(input_shapes, output_shape, [child_f, child_g])
        self.child_f = child_f
        self.child_g = child_g

    def forward(self, *xs):
        # assume shape: [..., D]
        return (self.child_f(*xs) * self.child_g(*xs)).sum(dim=-1)

    def set_submodule(self, index, new_module):
        if index == 0:
            self.child_f = new_module
        else:
            self.child_g = new_module

    def mutate(self, module_factory):
        if random.random() < 0.5:
            self.child_f = self.child_f.mutate(module_factory)
        if random.random() < 0.5:
            self.child_g = self.child_g.mutate(module_factory)
        return self

    @staticmethod
    def can_build(input_shapes, output_shape):
        # If all inputs are shaped [..., D], output should be [...] without D
        # Just a simple check: all inputs have same shape, last dim > 1, and output = same shape without last dim
        if len(input_shapes) >= 2:
            s = input_shapes[0]
            if all(shapes_equal(si, s) for si in input_shapes):
                if len(s) >= 1:
                    reduced_shape = s[:-1]
                    return reduced_shape == output_shape
        return False

    @staticmethod
    def infer_output_shape(input_shapes):
        s = input_shapes[0]
        return s[:-1]  # remove last dim

    def complexity(self):
        return 5.0

    def __str__(self):
        return f'dot({self.child_f},{self.child_g})'
