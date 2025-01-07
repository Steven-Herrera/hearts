"""Define and validate dataframe schemas."""

# %% IMPORTS

import typing as T

import pandas as pd
import pandera as pa
import pandera.typing as papd
import pandera.typing.common as padt

# %% TYPES

# Generic type for a dataframe container
TSchema = T.TypeVar("TSchema", bound="pa.DataFrameModel")

# %% SCHEMAS


class Schema(pa.DataFrameModel):
    """Base class for a dataframe schema.

    Use a schema to type your dataframe object.
    e.g., to communicate and validate its fields.
    """

    class Config:
        """Default configurations for all schemas.

        Parameters:
            coerce (bool): convert data type if possible.
            strict (bool): ensure the data type is correct.
        """

        coerce: bool = True
        strict: bool = True

    @classmethod
    def check(cls: T.Type[TSchema], data: pd.DataFrame) -> papd.DataFrame[TSchema]:
        """Check the dataframe with this schema.

        Args:
            data (pd.DataFrame): dataframe to check.

        Returns:
            papd.DataFrame[TSchema]: validated dataframe.
        """
        return T.cast(papd.DataFrame[TSchema], cls.validate(data))


class InputsSchema(Schema):
    """Schema for the project inputs."""

    padt
    Age: papd.Series[padt.UInt64] = pa.Field(ge=0)
    Sex: papd.Series[str] = pa.Field(isin=['M', 'F'])
    ChestPainType: papd.Series[str] = pa.Field(isin=['ATA' 'NAP' 'ASY' 'TA'])
    RestingBP: papd.Series[padt.UInt64] = pa.Field(gt=0)
    Cholesterol: papd.Series[padt.UInt64] = pa.Field(ge=0)  
    FastingBS: papd.Series[padt.UInt64] = pa.Field(isin=[0,1])
    RestingECG: papd.Series[str] = pa.Field(isin=['Normal' 'ST' 'LVH'])
    MaxHR: papd.Series[padt.UInt64] = pa.Field(gt=0, lt=220)
    ExerciseAngina: papd.Series[str] = pa.Field(isin=['Y', 'N'])
    Oldpeak: papd.Series[padt.Float64] = pa.Field()
    ST_Slope: papd.Series[str] = pa.Field(isin=['Up', 'Flat', 'Down'])


Inputs = papd.DataFrame[InputsSchema]


class TargetsSchema(Schema):
    """Schema for the project target."""

    HeartDisease: papd.Series[padt.UInt64] = pa.Field(isin=[0,1])


Targets = papd.DataFrame[TargetsSchema]


class OutputsSchema(Schema):
    """Schema for the project output."""

    prediction: papd.Series[padt.UInt32] = pa.Field(isin=[0,1])


Outputs = papd.DataFrame[OutputsSchema]


class SHAPValuesSchema(Schema):
    """Schema for the project shap values."""

    class Config:
        """Default configurations this schema.

        Parameters:
            dtype (str): dataframe default data type.
            strict (bool): ensure the data type is correct.
        """

        dtype: str = "float32"
        strict: bool = False


SHAPValues = papd.DataFrame[SHAPValuesSchema]


class FeatureImportancesSchema(Schema):
    """Schema for the project feature importances."""

    feature: papd.Series[padt.String] = pa.Field()
    importance: papd.Series[padt.Float32] = pa.Field()


FeatureImportances = papd.DataFrame[FeatureImportancesSchema]