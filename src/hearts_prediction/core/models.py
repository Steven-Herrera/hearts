"""Define trainable machine learning models."""

# %% IMPORTS

import abc

import pydantic as pdt
import shap
import typing_extensions as T
from sklearn import compose, pipeline, preprocessing
from xgboost.sklearn import XGBClassifier

from hearts_prediction.core import schemas

# %% TYPES

# Model params
ParamKey = str
ParamValue = T.Any
Params = dict[ParamKey, ParamValue]

# %% MODELS


class Model(abc.ABC, pdt.BaseModel, strict=True, frozen=False, extra="forbid"):
    """Base class for a project model.

    Use a model to adapt AI/ML frameworks.
    e.g., to swap easily one model with another.
    """

    KIND: str

    def get_params(self, deep: bool = True) -> Params:
        """Get the model params.

        Args:
            deep (bool, optional): ignored.

        Returns:
            Params: internal model parameters.
        """
        params: Params = {}
        for key, value in self.model_dump().items():
            if not key.startswith("_") and not key.isupper():
                params[key] = value
        return params

    def set_params(self, **params: ParamValue) -> T.Self:
        """Set the model params in place.

        Returns:
            T.Self: instance of the model.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    @abc.abstractmethod
    def fit(self, inputs: schemas.Inputs, targets: schemas.Targets) -> T.Self:
        """Fit the model on the given inputs and targets.

        Args:
            inputs (schemas.Inputs): model training inputs.
            targets (schemas.Targets): model training targets.

        Returns:
            T.Self: instance of the model.
        """

    @abc.abstractmethod
    def predict(self, inputs: schemas.Inputs) -> schemas.Outputs:
        """Generate outputs with the model for the given inputs.

        Args:
            inputs (schemas.Inputs): model prediction inputs.

        Returns:
            schemas.Outputs: model prediction outputs.
        """

    def explain_model(self) -> schemas.FeatureImportances:
        """Explain the internal model structure.

        Returns:
            schemas.FeatureImportances: feature importances.
        """
        raise NotImplementedError()

    def explain_samples(self, inputs: schemas.Inputs) -> schemas.SHAPValues:
        """Explain model outputs on input samples.

        Returns:
            schemas.SHAPValues: SHAP values.
        """
        raise NotImplementedError()

    def get_internal_model(self) -> T.Any:
        """Return the internal model in the object.

        Raises:
            NotImplementedError: method not implemented.

        Returns:
            T.Any: any internal model (either empty or fitted).
        """
        raise NotImplementedError()


class BaselineSklearnXGBModel(Model):
    """Simple baseline model based on scikit-learn.

    Parameters:
        max_depth (int): maximum depth of the random forest.
        n_estimators (int): number of estimators in the random forest.
        random_state (int, optional): random state of the machine learning pipeline.
    """

    KIND: T.Literal["BaselineSklearnXGBModel"] = "BaselineSklearnXGBModel"

    # params
    max_depth: int = 20
    n_estimators: int = 200
    objective: str = "binary:hinge"
    device: str = "cuda"
    verbosity: int = 2
    random_state: int | None = 42
    # private
    _pipeline: pipeline.Pipeline | None = None
    _numericals: list[str] = [
        "Age",
        "RestingBP",
        "Cholesterol",
        "MaxHR",
        "Oldpeak",
    ]
    _numerical_binary: list[str] = [
        "FastingBS",
    ]
    _categoricals: list[str] = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

    @T.override
    def fit(self, inputs: schemas.Inputs, targets: schemas.Targets) -> "BaselineSklearnXGBModel":
        # subcomponents
        categoricals_transformer = preprocessing.OneHotEncoder(
            sparse_output=False, handle_unknown="ignore", drop="if_binary"
        )
        numericals_transformer = preprocessing.Normalizer()

        # components
        transformer = compose.ColumnTransformer(
            [
                ("categoricals", categoricals_transformer, self._categoricals),
                ("numericals", numericals_transformer, self._numericals),
                ("numerical_binary", "passthrough", self._numerical_binary),
            ],
            remainder="drop",
        )
        clf = XGBClassifier(
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            objective=self.objective,
            device=self.device,
            verbosity=self.verbosity,
        )
        # pipeline
        self._pipeline = pipeline.Pipeline(
            steps=[
                ("transformer", transformer),
                ("classifier", clf),
            ]
        )
        self._pipeline.fit(X=inputs, y=targets[schemas.TargetsSchema.HeartDisease])
        return self

    @T.override
    def predict(self, inputs: schemas.Inputs) -> schemas.Outputs:
        model = self.get_internal_model()
        prediction = model.predict(inputs)
        outputs = schemas.Outputs(
            {schemas.OutputsSchema.prediction: prediction}, index=inputs.index
        )
        return outputs

    @T.override
    def explain_model(self) -> schemas.FeatureImportances:
        model = self.get_internal_model()
        clf = model.named_steps["classifier"]
        transformer = model.named_steps["transformer"]
        feature = transformer.get_feature_names_out()
        feature_importances = schemas.FeatureImportances(
            data={
                "feature": feature,
                "importance": clf.feature_importances_,
            }
        )
        return feature_importances

    @T.override
    def explain_samples(self, inputs: schemas.Inputs) -> schemas.SHAPValues:
        model = self.get_internal_model()
        classifier = model.named_steps["classifier"]
        transformer = model.named_steps["transformer"]
        transformed = transformer.transform(X=inputs)
        explainer = shap.TreeExplainer(model=classifier)
        shap_values = schemas.SHAPValues(
            data=explainer.shap_values(X=transformed),
            columns=transformer.get_feature_names_out(),
        )
        return shap_values

    @T.override
    def get_internal_model(self) -> pipeline.Pipeline:
        model = self._pipeline
        if model is None:
            raise ValueError("Model is not fitted yet!")
        return model


ModelKind = BaselineSklearnXGBModel
