"""High-level jobs of the project."""

# %% IMPORTS

from hearts_prediction.jobs.evaluations import EvaluationsJob
from hearts_prediction.jobs.explanations import ExplanationsJob
from hearts_prediction.jobs.inference import InferenceJob
from hearts_prediction.jobs.promotion import PromotionJob
from hearts_prediction.jobs.training import TrainingJob
from hearts_prediction.jobs.tuning import TuningJob

# %% TYPES

JobKind = TuningJob | TrainingJob | PromotionJob | InferenceJob | EvaluationsJob | ExplanationsJob

# %% EXPORTS

__all__ = [
    "TuningJob",
    "TrainingJob",
    "PromotionJob",
    "InferenceJob",
    "EvaluationsJob",
    "ExplanationsJob",
    "JobKind",
]
