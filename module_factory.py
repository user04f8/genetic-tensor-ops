import math
import random

from genetic_module import GeneticModuleMeta
from modules import (
    InputGM, ConstantGM, ParameterGM, ScalarParameterGM
)
from shape_utils import can_reduce

class ModuleFactory:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.leaf_modules = GeneticModuleMeta.registry['leaf']
        self.unary_modules = GeneticModuleMeta.registry['unary']
        self.binary_modules = GeneticModuleMeta.registry['binary']
        self.aggregate_modules = GeneticModuleMeta.registry['aggregate']
    def create_random_module(self, input_shapes, output_shape, max_attempts=10, depth=0):
        # If we are too deep, fallback to simple solutions
        if depth > self.max_depth:
            return self.fallback_module(input_shapes, output_shape)

        for _ in range(max_attempts):
            choice = random.random()
            if choice < 0.3:
                m = self.try_create_leaf(input_shapes, output_shape)
            elif choice < 0.6:
                m = self.try_create_unary(input_shapes, output_shape, depth)
            elif choice < 0.9:
                m = self.try_create_binary(input_shapes, output_shape, depth)
            else:
                m = self.try_create_aggregate(input_shapes, output_shape, depth)

            if m is not None:
                return m

        # fallback if no module created
        return self.fallback_module(input_shapes, output_shape)

    def fallback_module(self, input_shapes, output_shape):
        if ParameterGM.can_build(input_shapes, output_shape):
            return ParameterGM(input_shapes, output_shape)
        return ConstantGM(input_shapes, output_shape, 0.0)

    def try_create_leaf(self, input_shapes, output_shape):
        candidates = self.leaf_modules[:]
        random.shuffle(candidates)
        for c in candidates:
            if c.can_build(input_shapes, output_shape):
                if c == InputGM:
                    # pick a valid index
                    indices = [i for i, s in enumerate(input_shapes) if s == output_shape]
                    if indices:
                        return InputGM(input_shapes, output_shape, random.choice(indices))
                elif c == ConstantGM:
                    val = random.choice([0, 0.5,1,2,3,5,math.pi,math.e])
                    return ConstantGM(input_shapes, output_shape, val)
                elif c == ParameterGM:
                    return ParameterGM(input_shapes, output_shape)
                elif c == ScalarParameterGM and ScalarParameterGM.can_build(input_shapes, output_shape):
                    return ScalarParameterGM(input_shapes, output_shape)
        return None

    def try_create_unary(self, input_shapes, output_shape, depth):
        candidates = self.unary_modules[:]
        random.shuffle(candidates)
        for c in candidates:
            if c.can_build(input_shapes, output_shape):
                # need a child producing output_shape
                child = self.create_random_module(input_shapes, output_shape, depth=depth+1)
                if child is not None:
                    return c(input_shapes, output_shape, child)
        return None

    def try_create_binary(self, input_shapes, output_shape, depth):
        candidates = self.binary_modules[:]
        random.shuffle(candidates)
        for c in candidates:
            if c.can_build(input_shapes, output_shape):
                child_f = self.create_random_module(input_shapes, output_shape, depth=depth+1)
                if child_f is None:
                    continue
                child_g = self.create_random_module(input_shapes, output_shape, depth=depth+1)
                if child_g is None:
                    continue
                return c(input_shapes, output_shape, child_f, child_g)
        return None

    def try_create_aggregate(self, input_shapes, output_shape, depth):
        candidates = self.aggregate_modules[:]
        random.shuffle(candidates)
        for c in candidates:
            if c.can_build(input_shapes, output_shape):
                # try to find a suitable child shape
                # For simplicity, pick any input shape that can be reduced to output_shape
                possible_shapes = [s for s in input_shapes if can_reduce(s, output_shape)]
                if not possible_shapes:
                    continue
                s = random.choice(possible_shapes)
                child = self.create_random_module(input_shapes, s, depth=depth+1)
                if child is not None:
                    dim = 0 if len(s) > len(output_shape) else None
                    return c(input_shapes, output_shape, child, dim=dim)
        return None
