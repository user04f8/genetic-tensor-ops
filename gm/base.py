import torch
import torch.nn as nn
import random
from gm.genetic_module import GeneticModule, register_module
from shape_utils import can_select_input

@register_module('leaf')
class InputGM(GeneticModule):
    def __init__(self, input_shapes, output_shape, index=0):
        super().__init__(input_shapes, output_shape)
        self.index = index

    def forward(self, *xs):
        return xs[self.index]

    def mutate(self, module_factory):
        # Maybe choose a different input if available
        if len(self.input_shapes) > 1 and random.random() < 0.1:
            possible_indices = [i for i, s in enumerate(self.input_shapes) if s == self.output_shape]
            if possible_indices:
                self.index = random.choice(possible_indices)
        # small chance to replace module entirely
        if random.random() < 0.05:
            new_mod = module_factory.create_random_module(self.input_shapes, self.output_shape)
            if new_mod:
                return new_mod
        return self

    @staticmethod
    def can_build(input_shapes, output_shape):
        return can_select_input(input_shapes, output_shape)

    @staticmethod
    def infer_output_shape(input_shapes):
        # Without a chosen index, we can't infer uniquely. Factory must pick an index.
        return None

    def complexity(self):
        # InputGM is very simple: no parameters, just indexing
        return 1.0

    def __str__(self):
        return f'x{self.index}'

@register_module('leaf')
class ConstantGM(GeneticModule):
    def __init__(self, input_shapes, output_shape, value):
        super().__init__(input_shapes, output_shape)
        # value could be an integer, fraction, or known constant
        self.value = value
        # We'll store as a constant tensor
        self.register_buffer('const', torch.tensor(value).expand(output_shape))

    def forward(self, *xs):
        return self.const

    def mutate(self, module_factory):
        # Maybe slightly adjust the constant
        if isinstance(self.value, (int, float)):
            if random.random() < 0.1:
                self.value = self.value + random.uniform(-1,1)
                self.const = torch.tensor(self.value).expand(self.output_shape)
        if random.random() < 0.05:
            new_mod = module_factory.create_random_module(self.input_shapes, self.output_shape)
            if new_mod:
                return new_mod
        return self

    @staticmethod
    def can_build(input_shapes, output_shape):
        # Always can build a constant for any shape
        return True

    @staticmethod
    def infer_output_shape(input_shapes):
        # Need output_shape known
        return None

    def complexity(self):
        return 0.5
    
    def __str__(self):
        return f'{self.value:.4f}'

@register_module('leaf')
class ParameterGM(GeneticModule):
    def __init__(self, input_shapes, output_shape, init_strategy='uniform'):
        super().__init__(input_shapes, output_shape)
        self.param = nn.Parameter(torch.empty(output_shape))
        self.init_strategy = init_strategy
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_strategy == 'uniform':
            nn.init.uniform_(self.param, -0.1, 0.1)
        elif self.init_strategy == 'normal':
            nn.init.normal_(self.param, 0, 0.01)
        else:
            nn.init.constant_(self.param, 0.0)

    def forward(self, *xs):
        return self.param

    def mutate(self, module_factory):
        # Small chance to reinit
        if random.random() < 0.05:
            self.reset_parameters()
        # Chance to replace
        if random.random() < 0.05:
            new_mod = module_factory.create_random_module(self.input_shapes, self.output_shape)
            if new_mod:
                return new_mod
        return self

    @staticmethod
    def can_build(input_shapes, output_shape):
        return True

    @staticmethod
    def infer_output_shape(input_shapes):
        return None

    def complexity(self):
        # complexity = base + large penalty for each parameter
        num_params = self.param.numel()
        return 5.0 + num_params * 3.0  # big penalty for params

    def __str__(self):
        return f'Param<{self.init_strategy}>({list(self.param.shape)})'

@register_module('leaf')
class ScalarParameterGM(ParameterGM):
    def __init__(self, input_shapes, output_shape, init_strategy='constant'):
        # Force output_shape = ()
        super().__init__(input_shapes, (), init_strategy)

    @staticmethod
    def can_build(input_shapes, output_shape):
        # Only if output_shape == ()
        return len(output_shape) == 0

    def complexity(self):
        return 5.0

    def __str__(self):
        return f'ScalarParam<{self.init_strategy}>()'
