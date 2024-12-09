from gm.genetic_module import GeneticModule

class GeneticModuleOuter(GeneticModule):
    def __init__(self, input_shapes, output_shape, child):
        super().__init__(input_shapes, output_shape, [child])

    def forward(self, *xs):
        return self.child(*xs)