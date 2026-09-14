from .fno import TFNO, TFNO1d, TFNO2d, TFNO3d
from .fno import FNO, FNO1d, FNO2d, FNO3d
try:
    from .local_fno import LocalFNO
except (ModuleNotFoundError, ImportError):
    pass
# only import SFNO if torch_harmonics is built locally
try:
    from .sfno import SFNO
except (ModuleNotFoundError, ImportError):
    pass
from .uno import UNO
from .uqno import UQNO
from .fnogno import FNOGNO
from .gino import GINO
from .drainage_adapter import DrainageAdapter
from .base_model import get_model
