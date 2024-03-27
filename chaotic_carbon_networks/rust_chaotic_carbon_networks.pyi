import numpy as np
import numpy.typing as npt
from typing import Optional

def mind(
    x: npt.NDArray[np.float32], y: Optional[npt.NDArray[np.float32]], bins: int = 64
) -> npt.NDArray[np.float32]: ...
def lapend(
    x: npt.NDArray[np.float32], tau_min: int, tau_max: int, y: Optional[npt.NDArray[np.float32]]
) -> npt.NDArray[np.float32]: ...
