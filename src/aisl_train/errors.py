# SPDX-License-Identifier: Apache-2.0
"""Explicit failures. Do not convert these into warnings to continue."""

from __future__ import annotations


class AislTrainError(Exception):
    """User-visible runtime failure."""

    exit_code = 2


class ConfigError(AislTrainError):
    exit_code = 2


class UnsupportedModelError(AislTrainError):
    exit_code = 3


class MissingCapabilityError(AislTrainError):
    exit_code = 3


class MissingDependencyError(AislTrainError):
    exit_code = 3


class DatasetError(AislTrainError):
    exit_code = 2


class MaskingError(AislTrainError):
    exit_code = 2


class PeftError(AislTrainError):
    exit_code = 3


class CheckpointError(AislTrainError):
    exit_code = 2


class ExperimentError(AislTrainError):
    exit_code = 2


class EvaluationError(AislTrainError):
    exit_code = 2


class OutOfMemoryError(AislTrainError):
    exit_code = 4


class TrainingError(AislTrainError):
    exit_code = 4
