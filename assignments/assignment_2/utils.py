import numpy.typing as npt
import numpy as np
from settings import settings



RNG = np.random.default_rng(settings.SEED)

def make_random_weights(
    input_size: int,
    output_size: int,
) -> list[npt.NDArray[np.float64]]:
    """Draw a random parameter set for `nn_controller`.

    THIS IS THE FUNCTION YOUR EA REPLACES. Instead of sampling weights from a
    normal distribution, your EA will search for them.

    Note the total parameter count printed by main(): that is the length of the
    flat vector an individual's genotype has to encode. Reshaping a flat
    genotype back into these matrices is on you.
    """
    return [
        RNG.normal(scale=0.5, size=(input_size, settings.HIDDEN_SIZE)),
        RNG.normal(scale=0.5, size=(settings.HIDDEN_SIZE, output_size)),
    ]
