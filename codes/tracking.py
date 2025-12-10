import weave
import wandb
from abc import ABC, abstractmethod
from wandb_utils import initialize_wandb_run, upload_artifact

# --- The Abstract Base Class (The "Contract") ---
class AttrDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__

class AbstractExperimentTracker(ABC):
    """
    An abstract base class that defines the contract for all experiment trackers.
    """
    @abstractmethod
    def __init__(self, config):
        """Initialize the tracker with the given configuration."""
        pass
    
    def trace(self, func):
        """A method that can act as a decorator for tracing function calls."""
        return func

    @abstractmethod
    def log_metrics(self, data_dict, step=None):
        """Log a dictionary of metrics."""
        pass

    @abstractmethod
    def save_artifact(self, **kwargs):
        """Save an artifact."""
        pass

# --- The Original ExperimentTracker, Now Implementing the Abstract Class ---

class ExperimentTracker(AbstractExperimentTracker):
    """
    A generic tracker that logs all events to the console.
    This is the default for running without third-party services.
    """
    def __init__(self, config):
        print("[TRACKING - INIT]: Console-based ExperimentTracker initialized.")
    
    def log_metrics(self, data_dict, step=None):
        """Prints metrics to the console."""
        print(f"[TRACKING - LOG]: {data_dict}")

    def save_artifact(self, **kwargs):
        """Prints artifact-saving information to the console."""
        print(f"[TRACKING - ARTIFACT]: Saving artifact with args: {kwargs}")


# --- The W&B and Weave Implementation, Also Implementing the Abstract Class ---

class WandbWeaveTracker(AbstractExperimentTracker):
    """An implementation of the tracker for Weights & Biases and Weave."""
    def __init__(self, config):
        print(f"{config}")
        self.run_id = initialize_wandb_run(
            id=config.wandb_run_id, 
            project="nsgf-paper2code-playground", 
            entity="rogermt23", 
            stage="evaluation"
        )
        if self.run_id is None:
            raise RuntimeError(f"Failed to initialize W&B run with ID {config.wandb_run_id}")

    def trace(self, func):
        """Applies the @weave.op() decorator."""
        return weave.op()(func)

    def log_metrics(self, data_dict, step=None):
        """Log metrics to W&B."""
        wandb.log(data_dict, step=step)

    def save_artifact(self, **kwargs):
        """Save an artifact by calling the imported utility function."""
        upload_artifact(**kwargs)