def shapes_equal(s1, s2):
    return s1 == s2

def can_elementwise(input_shapes, output_shape):
    # Elementwise ops require all input shapes to match output shape
    return all(s == output_shape for s in input_shapes)

def can_select_input(input_shapes, output_shape):
    # If we want to pick one input that matches output_shape
    return any(shapes_equal(s, output_shape) for s in input_shapes)

def can_reduce(input_shape, output_shape):
    # Check if output_shape is a reduced form of input_shape by removing or reducing some dims
    # A simple heuristic: output_shape must be <= input_shape in rank and each dimension matches
    # except for reduced ones.
    # For simplicity, assume output_shape is a strict subset of input_shape with some dims summed out.
    if len(output_shape) <= len(input_shape):
        # We must find a sequence of sums that turn input_shape into output_shape
        # For simplicity, let's require output_shape be exactly input_shape with some dims removed.
        # More complex logic could be implemented.
        # Check if output_shape can be obtained by removing some dimensions of size > 1 from input_shape.
        # Not perfect, but works as an example.
        # If output_shape == input_shape, it's also valid if we sum over a dimension of size 1.
        # We'll just allow equality as trivial no-op sum or sum over dim of size=1.
        if output_shape == input_shape:
            return True
        # Try to see if output_shape is input_shape with one dimension removed:
        # This is a simplified assumption.
        for dim in range(len(input_shape)):
            # Remove dim
            reduced = input_shape[:dim] + input_shape[dim+1:]
            if reduced == output_shape:
                return True
    return False
