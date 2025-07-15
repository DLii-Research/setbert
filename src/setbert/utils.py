from numba import njit
import numpy as np
import numpy.typing as npt
from typing import Generator, List, Optional, Tuple

@njit
def build_tree(cdf: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    l: int
    r: int
    m: int
    left: np.float64
    output = np.empty(len(cdf) - 1, dtype=np.float64)
    stack: List[Tuple[int, int]] = [(0, len(output))]
    while len(stack) > 0:
        l, r = stack.pop()
        m = (r + l)//2
        left = cdf[l - 1] if l > 0 else np.float64(0.0)
        if cdf[r] - left == 0.0:
            output[m] = np.inf
        else:
            output[m] = (cdf[m] - left) / (cdf[r] - left)
        if m > l:
            stack.append((l, m))
        if m + 1< r:
            stack.append((m+1, r))
    return output

def sample(
    n: int,
    abundance_sampling_tree: npt.NDArray[np.float64],
    rng: Optional[np.random.Generator] = None
) -> Generator[Tuple[int, int], None, None]:
    """
    Sample n items from the tree.
    """
    rng = rng if rng is not None else np.random.default_rng()
    stack = [(n, 0, len(abundance_sampling_tree))]
    while len(stack) > 0:
        n, l, r = stack.pop()
        if l >= r:
            yield (l, n) # (index, count)
            continue
        m = (l + r)//2
        n_left = rng.binomial(n, abundance_sampling_tree[m])
        n_right = n - n_left
        if n_right > 0:
            stack.append((n_right, m+1, r))
        if n_left > 0:
            stack.append((n_left, l, m))
