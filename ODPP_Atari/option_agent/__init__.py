from .VIC import VIC_Agent
from .DIAYN import DIAYN_Agent
from .VALOR import VALOR_Agent
from .ODPP import ODPP_Agent
from .DCO import DCO_Agent

REGISTRY = {}
REGISTRY["VIC"] = VIC_Agent
REGISTRY["DIAYN"] = DIAYN_Agent
REGISTRY["VALOR"] = VALOR_Agent
REGISTRY["ODPP"] = ODPP_Agent
REGISTRY["DCO"] = DCO_Agent

