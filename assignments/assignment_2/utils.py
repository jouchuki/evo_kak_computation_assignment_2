from assignment_types import Genotype, Weights, WeightShapes
import numpy as np
from settings import settings




def make_random_weights(
    input_size: int,
    output_size: int,
    rng: np.random.Generator,
    *,
    hidden_size: int = settings.HIDDEN_SIZE,
) -> Weights:
    """Draw a random parameter set for `nn_controller`.

    THIS IS THE FUNCTION YOUR EA REPLACES. Instead of sampling weights from a
    normal distribution, your EA will search for them.

    Note the total parameter count printed by main(): that is the length of the
    flat vector an individual's genotype has to encode. Reshaping a flat
    genotype back into these matrices is on you.
    """
    return [
        rng.normal(scale=0.5, size=(input_size, hidden_size)),
        rng.normal(scale=0.5, size=(hidden_size, output_size)),
    ]


def flatten_weights(weights: Weights) -> Genotype:
    return np.concatenate([w.ravel() for w in weights]).tolist()

# ... to describe any number of integers
def unflatten_weights(genes: Genotype, shapes: WeightShapes) -> Weights:
    flat = np.asarray(genes, dtype=np.float64)
    sizes = [int(np.prod(shape)) for shape in shapes]

    if flat.ndim != 1 or flat.size != sum(sizes):
        raise ValueError("Genotype doesn't match the expected weight shapes")

    weights = []
    start = 0

    for shape, size in zip(shapes, sizes):
        weights.append(flat[start:start + size].reshape(shape).copy())
        start += size

    return weights