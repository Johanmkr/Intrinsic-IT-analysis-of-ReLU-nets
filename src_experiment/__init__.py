# __all__ = [
#     "get_args", 
#     "NeuralNet", "moons_models", "paths", "createfolders", "train_model", "get_specific_moon_state_dict", "get_df_of_run_summary", "get_df_of_convergence_summary", "get_moon_dataloaders", "get_new_moons_data", "get_new_moons_data_for_all_noises", "get_model", "VisualisationOfLatenRepresentation"]

# from .arg_parser import get_args
# from .utils import NeuralNet, createfolders, get_specific_moon_state_dict, get_df_of_run_summary, get_df_of_convergence_summary
# from .paths import *
# from .dataset import get_new_moons_data, get_new_moons_data_for_all_noises

# from .train_models import train_model  
# from .models import moons_models, get_model
# from .visualize_data_representation import VisualisationOfLatenRepresentation



# mypackage/__init__.py
import importlib

# Mapping of public names to their source modules
_lazy_attributes = {
    "get_moons_data": ".dataset",
    "get_wbc_data": ".dataset",
    "get_wine_data": ".dataset",
    "get_hd_data": ".dataset",
    "get_car_data": ".dataset",
    "get_new_data": ".dataset",
    "QUANTITIES_TO_ESTIMATE": ".divergence_engine",
    "DivergenceEngine": ".divergence_engine",
    "EstimateQuantities1Run": ".estimate_quantities",
    "AverageEstimates": ".estimate_quantities",
    "get_model": ".models",
    "moon_path": ".paths",
    "wbc_path": ".paths",
    "get_new_path": ".paths",
    "get_test_data": ".paths",
    "neurips_figpath": ".pahts",
    "train_model": ".train_models",
    "train_model_multiclass": ".train_models",
    "NeuralNet": ".utils",
    "createfolders": ".utils",
    "savefig": ".utils",
}

# Dynamically define __all__ based on lazy attributes
__all__ = list(_lazy_attributes.keys())



def __getattr__(name):
    if name in _lazy_attributes:
        module = importlib.import_module(
            _lazy_attributes[name],
            package=__name__
        )
        value = getattr(module, name)
        globals()[name] = value  # cache
        return value
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    """
    Include lazily-loaded attributes in dir() calls.
    """
    return sorted(list(globals().keys()) + list(_lazy_attributes.keys()))
