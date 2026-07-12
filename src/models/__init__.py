from src.models.gbm import CatWrap, LGBMWrap, TabularModel, XGBWrap

__all__ = ["LGBMWrap", "XGBWrap", "CatWrap", "TabularModel"]

# ---- FT-Transformer: prefer the rtdl + torch-directml version if available ----
# The rtdl path bypasses Lightning entirely so DirectML (or CPU) works
# without segfaults. The pytorch-tabular version stays around as a fallback
# for TabNet / NODE only.
FTTransformerWrap = None
try:
    from src.models.dl_rtdl import FTTransformerWrap as _FTTransformerRtdl  # noqa
    FTTransformerWrap = _FTTransformerRtdl
    __all__.append("FTTransformerWrap")
    RTDL_AVAILABLE = True
except ImportError:
    RTDL_AVAILABLE = False

# TabNet / NODE still require pytorch-tabular (no equivalent maintained
# non-Lightning reference implementations at time of writing).
try:
    from src.models.dl import NodeWrap, TabNetWrap  # noqa
    if FTTransformerWrap is None:
        from src.models.dl import FTTransformerWrap as _FTTransformerPT  # noqa
        FTTransformerWrap = _FTTransformerPT
        __all__.append("FTTransformerWrap")
    __all__ += ["TabNetWrap", "NodeWrap"]
    PTTAB_AVAILABLE = True
except ImportError:
    PTTAB_AVAILABLE = False

DL_AVAILABLE = RTDL_AVAILABLE or PTTAB_AVAILABLE


def registry() -> dict:
    """Name -> class map. DL entries only present if their libs are installed."""
    reg: dict[str, type[TabularModel]] = {
        "lightgbm": LGBMWrap,
        "xgboost": XGBWrap,
        "catboost": CatWrap,
    }
    if FTTransformerWrap is not None:
        reg["ft_transformer"] = FTTransformerWrap
    if PTTAB_AVAILABLE:
        reg["tabnet"] = TabNetWrap
        reg["node"] = NodeWrap
    return reg
