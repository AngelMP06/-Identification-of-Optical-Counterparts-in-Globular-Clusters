from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray


@dataclass

class FilterParams:
    # Filters are defined to remove low-quality data without being too restrictive. 
    Mag_min: float = -50
    Mag_max: float = 50
    RMS_max: float = 1
    Fit_max: float = 1.5
    Sharp_min: float = -2
    Sharp_max: float = 2
    CM_min: float = 80

@dataclass
class MSDetectionParams:
    min_count: int = 2
    min_step: int = 2
    maximum_lenght_MSTO: float = 0.70

@dataclass
class MainSequenceResults:
    # Main Sequence y-values (magnitudes) along the sequence
    main_sequence_y: NDArray[np.float64]
    # Smoothed left boundary of the Main Sequence
    min_smooth: NDArray[np.float64]
    # Smoothed right boundary of the Main Sequence
    max_smooth: NDArray[np.float64]
    # Index marking the start of the Subgiant Branch (SGB)
    index_SGB: int = 0
    # Index of the Main Sequence Turn-Off (MSTO)
    index_MSTO: int = 0
    # Value of the y-axis (magnitude) at the start of the Main Sequence Turn-Off (MSTO)
    msto_y: float = 0.0

@dataclass
class Points:
    # Key points used for classification
    
    point_A: NDArray[np.float64] = field(default_factory=lambda: np.zeros(2))
    point_B: NDArray[np.float64] = field(default_factory=lambda: np.zeros(2))
    point_C: NDArray[np.float64] = field(default_factory=lambda: np.zeros(2))
    point_D: NDArray[np.float64] = field(default_factory=lambda: np.zeros(2))
    point_E: NDArray[np.float64] = field(default_factory=lambda: np.zeros(2))
    point_F: NDArray[np.float64] = field(default_factory=lambda: np.zeros(2))
    point_G: NDArray[np.float64] = field(default_factory=lambda: np.zeros(2))
    point_H: NDArray[np.float64] = field(default_factory=lambda: np.zeros(2))