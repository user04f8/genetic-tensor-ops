import torch
import torch.nn as nn
import random
from genetic_module import GeneticModule

class IdentityGM(GeneticModule):
    def __init__(self, input_shapes, output_shape):
        super().__init__(input_shapes, output_shape)
        # Since identity: must have exactly one input and that input matches output
        # If not matched, fallback handled in factory.

    def forward(self, x):
        return x

    def mutate(self, module_factory):
        # 10% chance to replace entirely
        if random.random() < 0.1:
            return module_factory.create_random_module(self.input_shapes, self.output_shape)
        return self

    @staticmethod
    def can_build(input_shapes, output_shape):
        return len(input_shapes) == 1 and input_shapes[0] == output_shape

    @staticmethod
    def infer_output_shape(input_shapes):
        if len(input_shapes) == 1:
            return input_shapes[0]
        return None
    
    def __str__(self):
        return f'(x)'

class ConstantGM(GeneticModule):
    def __init__(self, input_shapes, output_shape):
        super().__init__(input_shapes, output_shape)
        self.const = nn.Parameter(torch.zeros(output_shape), requires_grad=False)

    def forward(self, *xs):
        return self.const

    def mutate(self, module_factory):
        if random.random() < 0.05:
            with torch.no_grad():
                self.const += 0.1 * torch.randn_like(self.const)
        if random.random() < 0.05:
            return module_factory.create_random_module(self.input_shapes, self.output_shape)
        return self

    @staticmethod
    def can_build(input_shapes, output_shape):
        # Always can build a constant for any shape
        return True

    @staticmethod
    def infer_output_shape(input_shapes):
        # For constants, we must be given an output shape.
        # If called without a known output shape, return None.
        return None
    
    def __str__(self):
        return str(self.const)

class ParameterGM(GeneticModule):
    def __init__(self, input_shapes, output_shape):
        super().__init__(input_shapes, output_shape)
        self.param = nn.Parameter(torch.randn(output_shape)*0.01)

    def forward(self, *xs):
        return self.param

    def mutate(self, module_factory):
        if random.random() < 0.05:
            with torch.no_grad():
                self.param.data = torch.randn(self.param.shape)*0.01
        if random.random() < 0.05:
            return module_factory.create_random_module(self.input_shapes, self.output_shape)
        return self

    @staticmethod
    def can_build(input_shapes, output_shape):
        # Any shape can have a parameter
        return True

    @staticmethod
    def infer_output_shape(input_shapes):
        # Same as ConstantGM, needs output shape known in advance
        return None
    
    def __str__(self):
        return str(self.param)

class VariableGM(GeneticModule):
    def __init__(self, input_shapes, output_shape, index=0):
        super().__init__(input_shapes, output_shape)
        self.index = index

    def forward(self, *xs):
        return xs[self.index]

    def mutate(self, module_factory):
        if len(self.input_shapes) > 1 and random.random() < 0.1:
            # Try changing index to another valid input that matches output_shape
            valid_indices = [i for i, s in enumerate(self.input_shapes) if s == self.output_shape]
            if valid_indices:
                self.index = random.choice(valid_indices)
        if random.random() < 0.05:
            return module_factory.create_random_module(self.input_shapes, self.output_shape)
        return self

    @staticmethod
    def can_build(input_shapes, output_shape):
        # Must have at least one input that matches output shape
        return any(s == output_shape for s in input_shapes)

    @staticmethod
    def infer_output_shape(input_shapes):
        # If used blindly, we need a chosen input. Without it, no unique inference is possible.
        # Typically, factory chooses after confirming a match.
        return None
    
    def __str__(self):
        return f'x{self.index + 1}'

class AddGM(GeneticModule):
    def __init__(self, input_shapes, output_shape, child_f, child_g):
        super().__init__(input_shapes, output_shape)
        self.child_f = child_f
        self.child_g = child_g

    def forward(self, *xs):
        return self.child_f(*xs) + self.child_g(*xs)

    def get_submodules(self):
        return [self.child_f, self.child_g]

    def set_submodule(self, index, new_module):
        if index == 0:
            self.child_f = new_module
        elif index == 1:
            self.child_g = new_module

    def mutate(self, module_factory):
        if random.random() < 0.5:
            self.child_f = self.child_f.mutate(module_factory)
        if random.random() < 0.5:
            self.child_g = self.child_g.mutate(module_factory)
        if random.random() < 0.05:
            # try replacing a child with a valid module
            candidate = module_factory.create_random_module(self.child_f.input_shapes, self.child_f.output_shape)
            if candidate is not None:
                self.child_f = candidate
        if random.random() < 0.05:
            candidate = module_factory.create_random_module(self.child_g.input_shapes, self.child_g.output_shape)
            if candidate is not None:
                self.child_g = candidate
        if random.random() < 0.01:
            candidate = module_factory.create_random_module(self.input_shapes, self.output_shape)
            if candidate is not None:
                return candidate
        return self

    @staticmethod
    def can_build(input_shapes, output_shape):
        # For AddGM, we need two children with the same output shape = output_shape
        # We'll rely on the factory to find suitable children. If it can, great.
        return True

    @staticmethod
    def infer_output_shape(input_shapes):
        # Add does not inherently change shape, it requires two children that produce the same shape.
        # The output_shape must be known or set by factory after choosing children.
        return None
    
    def __str__(self):
        return f'{self.child_f} + {self.child_g}'

class MultGM(GeneticModule):
    def __init__(self, input_shapes, output_shape, child_f, child_g):
        super().__init__(input_shapes, output_shape)
        self.child_f = child_f
        self.child_g = child_g

    def forward(self, *xs):
        return self.child_f(*xs) * self.child_g(*xs)

    def get_submodules(self):
        return [self.child_f, self.child_g]

    def set_submodule(self, index, new_module):
        if index == 0:
            self.child_f = new_module
        elif index == 1:
            self.child_g = new_module

    def mutate(self, module_factory):
        if random.random() < 0.5:
            self.child_f = self.child_f.mutate(module_factory)
        if random.random() < 0.5:
            self.child_g = self.child_g.mutate(module_factory)
        if random.random() < 0.05:
            candidate = module_factory.create_random_module(self.child_f.input_shapes, self.child_f.output_shape)
            if candidate is not None:
                self.child_f = candidate
        if random.random() < 0.05:
            candidate = module_factory.create_random_module(self.child_g.input_shapes, self.child_g.output_shape)
            if candidate is not None:
                self.child_g = candidate
        if random.random() < 0.01:
            candidate = module_factory.create_random_module(self.input_shapes, self.output_shape)
            if candidate is not None:
                return candidate
        return self

    @staticmethod
    def can_build(input_shapes, output_shape):
        # Similar to AddGM
        return True

    @staticmethod
    def infer_output_shape(input_shapes):
        return None
    
    def __str__(self):
        return f'{self.child_f}*{self.child_g}'

class SquareGM(GeneticModule):
    def __init__(self, input_shapes, output_shape, child_f):
        super().__init__(input_shapes, output_shape)
        self.child_f = child_f

    def forward(self, *xs):
        return self.child_f(*xs)**2

    def get_submodules(self):
        return [self.child_f]

    def set_submodule(self, index, new_module):
        if index == 0:
            self.child_f = new_module

    def mutate(self, module_factory):
        if random.random() < 0.5:
            self.child_f = self.child_f.mutate(module_factory)
        if random.random() < 0.01:
            candidate = module_factory.create_random_module(self.input_shapes, self.output_shape)
            if candidate is not None:
                return candidate
        return self

    @staticmethod
    def can_build(input_shapes, output_shape):
        # Square just applies element-wise op, child_f must have output_shape = desired output_shape
        return True

    @staticmethod
    def infer_output_shape(input_shapes):
        # If we know the child must produce output_shape = the final output
        # We'll rely on the factory to set it.
        return None
    
    def __str__(self):
        return f'{self.child_f}**2'

class SumGM(GeneticModule):
    def __init__(self, input_shapes, output_shape, child_f, dim=None):
        super().__init__(input_shapes, output_shape)
        self.child_f = child_f
        self.dim = dim

    def forward(self, *xs):
        out = self.child_f(*xs)
        # sum along dim
        return out.sum(dim=self.dim, keepdim=(self.output_shape == out.sum(dim=self.dim, keepdim=True).shape))

    def get_submodules(self):
        return [self.child_f]

    def set_submodule(self, index, new_module):
        if index == 0:
            self.child_f = new_module

    def mutate(self, module_factory):
        # Maybe change dim if possible
        if random.random() < 0.1 and len(self.output_shape) < len(self.child_f.output_shape):
            # Try a different dim that is valid
            candidate_dims = range(len(self.child_f.output_shape))
            self.dim = random.choice(list(candidate_dims))
        if random.random() < 0.5:
            self.child_f = self.child_f.mutate(module_factory)
        return self

    @staticmethod
    def can_build(input_shapes, output_shape):
        # We must find a child that can produce a shape that can be reduced to output_shape by summation.
        # For simplicity, let's say SumGM reduces exactly one dimension:
        # The child_f output shape should be identical to output_shape except for one dimension that can be summed out.
        # If output_shape == child_f_outshape after summation, we can handle it.
        return True

    @staticmethod
    def infer_output_shape(input_shapes):
        # Without specifying dim or child's shape, we can't infer. Factory must handle carefully.
        return None
    
    def __str__(self):
        return f'sum({self.child_f}, dim={self.dim})'
