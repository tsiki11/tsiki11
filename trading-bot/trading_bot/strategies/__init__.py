from .donchian_4h import Donchian4H
from .ma_cross_4h import MACross4H
from .ma_cross_baseline import MACrossBaseline

REGISTRY = {"S001": MACrossBaseline, "S002": MACross4H, "S003": Donchian4H}
