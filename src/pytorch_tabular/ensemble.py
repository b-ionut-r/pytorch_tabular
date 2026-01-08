# Pytorch Tabular - Multi-Config Ensemble
# Extension for running multiple model configurations and ensembling predictions
"""Multi-Config Ensemble for Deep Learning AutoML."""
from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.base import TransformerMixin

# Avoid circular import - import inside methods or use TYPE_CHECKING
if TYPE_CHECKING:
    from pytorch_tabular import TabularModel
    from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig

logger = logging.getLogger(__name__)


class MultiConfigEnsemble:
    """
    Run multiple pre-defined model configurations and ensemble their predictions.

    This class provides an AutoML-like experience by training multiple deep learning
    architectures with different hyperparameters and combining their predictions
    using weighted averaging based on cross-validation scores.

    Example:
        >>> from pytorch_tabular.ensemble import MultiConfigEnsemble
        >>> from pytorch_tabular.models import TabNetModelConfig, FTTransformerConfig
        >>>
        >>> configs = [
        ...     ("TabNet", TabNetModelConfig(task="regression", ...)),
        ...     ("FTTransformer", FTTransformerConfig(task="regression", ...)),
        ... ]
        >>> ensemble = MultiConfigEnsemble(configs, data_config, trainer_config)
        >>> ensemble.fit(train_data, cv=3, groups="survey_id", metric=custom_metric)
        >>> predictions = ensemble.predict(test_data)
    """

    def __init__(
        self,
        model_configs: List[Tuple[str, Any]],
        data_config: DataConfig,
        trainer_config: Optional[TrainerConfig] = None,
        optimizer_config: Optional[OptimizerConfig] = None,
        verbose: bool = True,
    ):
        """
        Initialize the MultiConfigEnsemble.

        Args:
            model_configs: List of (name, ModelConfig) tuples. Each ModelConfig
                should be a valid pytorch_tabular model configuration like
                TabNetModelConfig, FTTransformerConfig, NodeConfig, etc.
            data_config: DataConfig specifying target, categorical/continuous columns.
            trainer_config: Optional TrainerConfig for training parameters.
            optimizer_config: Optional OptimizerConfig for optimizer settings.
            verbose: If True, log progress information.
        """
        # Local imports to avoid circular dependency
        from pytorch_tabular.config import OptimizerConfig, TrainerConfig

        self.model_configs = model_configs
        self.data_config = data_config
        self.trainer_config = trainer_config or TrainerConfig(
            auto_lr_find=False,
            max_epochs=100,
            early_stopping="valid_loss",
            early_stopping_patience=10,
            progress_bar="simple",  # Avoid Rich progress bar crash with multiple models
        )
        self.optimizer_config = optimizer_config or OptimizerConfig()
        self.verbose = verbose

        # Will be populated during fit
        self.fitted_models: List[Tuple[str, Any, float]] = []  # (name, model, score)
        self.weights: Optional[np.ndarray] = None

    def fit(
        self,
        train: DataFrame,
        cv: int = 3,
        groups: Optional[Union[str, np.ndarray]] = None,
        metric: Optional[Callable] = None,
        feature_generator: Optional[TransformerMixin] = None,
        reset_datamodule: bool = True,
        **kwargs,
    ) -> "MultiConfigEnsemble":
        """
        Fit all model configurations using cross-validation.

        Args:
            train: Training data with features and target.
            cv: Number of cross-validation folds.
            groups: Group labels for GroupKFold. Can be column name or array.
            metric: Callable metric function(y_true, y_pred) -> score.
                Lower scores are assumed to be better (like loss).
            feature_generator: Optional transformer to apply before encoding.
            reset_datamodule: If True, reset datamodule for each fold.
            **kwargs: Additional arguments passed to TabularModel.cross_validate().

        Returns:
            self
        """
        # Local import to avoid circular dependency
        from pytorch_tabular import TabularModel

        self.fitted_models = []

        for name, model_config in self.model_configs:
            if self.verbose:
                logger.info(f"\n{'='*50}")
                logger.info(f"Training model: {name}")
                logger.info(f"{'='*50}")

            try:
                # Create fresh TabularModel for this config
                model = TabularModel(
                    data_config=self.data_config,
                    model_config=model_config,
                    optimizer_config=self.optimizer_config,
                    trainer_config=copy.deepcopy(self.trainer_config),
                    verbose=self.verbose,
                )

                # Run cross-validation
                cv_scores, oof_preds = model.cross_validate(
                    cv=cv,
                    train=train,
                    groups=groups,
                    metric=metric,
                    feature_generator=feature_generator,
                    reset_datamodule=reset_datamodule,
                    return_oof=True,
                    verbose=self.verbose,
                    **kwargs,
                )

                # Calculate mean CV score
                mean_score = np.mean(cv_scores)
                std_score = np.std(cv_scores)

                if self.verbose:
                    logger.info(f"{name} CV Score: {mean_score:.4f} (+/- {std_score:.4f})")

                self.fitted_models.append((name, model, mean_score))

            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                if self.verbose:
                    import traceback
                    traceback.print_exc()
                continue

        # Calculate ensemble weights (lower score = higher weight, assuming loss metric)
        if self.fitted_models:
            scores = np.array([s for _, _, s in self.fitted_models])
            # Convert to weights: lower loss = higher weight
            # Use softmax on negative scores
            self.weights = self._softmax(-scores)

            if self.verbose:
                logger.info(f"\n{'='*50}")
                logger.info("Ensemble Summary")
                logger.info(f"{'='*50}")
                for (name, _, score), weight in zip(self.fitted_models, self.weights):
                    logger.info(f"{name}: score={score:.4f}, weight={weight:.4f}")

        return self

    def _softmax(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Compute softmax values for x."""
        x = x / temperature
        e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return e_x / e_x.sum()

    def predict(
        self,
        test: DataFrame,
        feature_generator: Optional[TransformerMixin] = None,
        aggregate: str = "weighted_mean",
    ) -> DataFrame:
        """
        Generate ensemble predictions on test data.

        Args:
            test: Test data with features.
            feature_generator: If provided, transform test data before prediction.
                Should be the same fitted generator used during training.
            aggregate: Aggregation method. Options:
                - "weighted_mean": Weighted average using CV-based weights (default)
                - "mean": Simple average
                - "median": Median of predictions

        Returns:
            DataFrame with predictions.
        """
        if not self.fitted_models:
            raise RuntimeError("No models fitted. Call fit() first.")

        # Apply feature generator if provided
        test_transformed = test
        if feature_generator is not None:
            # For prediction, we only transform (generator was fit during CV)
            target_col = self.data_config.target[0] if isinstance(
                self.data_config.target, list
            ) else self.data_config.target

            # If target is in test data, separate it
            if target_col in test.columns:
                y_test = test[target_col]
                feature_cols = [c for c in test.columns if c != target_col]
                X_test = test[feature_cols]
                X_test_transformed = feature_generator.transform(X_test)
                if isinstance(X_test_transformed, np.ndarray):
                    X_test_transformed = pd.DataFrame(
                        X_test_transformed, index=X_test.index
                    )
                # Handle NaN/inf in numeric columns
                numeric_cols = X_test_transformed.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    X_test_transformed[col] = X_test_transformed[col].replace([np.inf, -np.inf], np.nan)
                    fill_val = X_test_transformed[col].median()
                    if pd.isna(fill_val):
                        fill_val = 0
                    X_test_transformed[col] = X_test_transformed[col].fillna(fill_val)
                test_transformed = pd.concat([X_test_transformed, y_test], axis=1)
            else:
                X_test_transformed = feature_generator.transform(test)
                if isinstance(X_test_transformed, np.ndarray):
                    X_test_transformed = pd.DataFrame(
                        X_test_transformed, index=test.index
                    )
                test_transformed = X_test_transformed

            # Handle NaN/inf in numeric columns (same as cross_validate)
            numeric_cols = test_transformed.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                test_transformed[col] = test_transformed[col].replace([np.inf, -np.inf], np.nan)
                fill_val = test_transformed[col].median()
                if pd.isna(fill_val):
                    fill_val = 0
                test_transformed[col] = test_transformed[col].fillna(fill_val)

        # Get predictions from all models
        all_predictions = []
        for name, model, _ in self.fitted_models:
            if self.verbose:
                logger.info(f"Predicting with {name}...")
            try:
                pred = model.predict(test_transformed)
                all_predictions.append(pred)
            except Exception as e:
                logger.error(f"Error predicting with {name}: {e}")
                continue

        if not all_predictions:
            raise RuntimeError("No successful predictions from any model.")

        # Stack predictions
        # Assuming regression with single target column
        pred_values = np.array([
            p.values if hasattr(p, 'values') else p
            for p in all_predictions
        ])

        # Aggregate predictions
        if aggregate == "weighted_mean":
            # Use weights calculated from CV scores
            valid_weights = self.weights[:len(pred_values)]
            valid_weights = valid_weights / valid_weights.sum()  # Renormalize
            ensemble_pred = np.average(pred_values, axis=0, weights=valid_weights)
        elif aggregate == "mean":
            ensemble_pred = np.mean(pred_values, axis=0)
        elif aggregate == "median":
            ensemble_pred = np.median(pred_values, axis=0)
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate}")

        # Create output DataFrame
        result = pd.DataFrame(
            ensemble_pred,
            index=test.index,
            columns=all_predictions[0].columns if hasattr(all_predictions[0], 'columns') else None,
        )

        return result

    def get_leaderboard(self) -> DataFrame:
        """
        Get a leaderboard of all fitted models.

        Returns:
            DataFrame with model names, scores, and weights.
        """
        if not self.fitted_models:
            return pd.DataFrame(columns=["model", "cv_score", "weight"])

        return pd.DataFrame({
            "model": [name for name, _, _ in self.fitted_models],
            "cv_score": [score for _, _, score in self.fitted_models],
            "weight": self.weights[:len(self.fitted_models)] if self.weights is not None else None,
        }).sort_values("cv_score")

    def refit_best(
        self,
        train: DataFrame,
        feature_generator: Optional[TransformerMixin] = None,
        top_k: int = 1,
    ) -> "MultiConfigEnsemble":
        """
        Refit the top-k models on the full training data.

        Args:
            train: Full training data.
            feature_generator: Optional transformer.
            top_k: Number of top models to refit.

        Returns:
            self
        """
        if not self.fitted_models:
            raise RuntimeError("No models fitted. Call fit() first.")

        # Sort by score (ascending, assuming lower is better)
        sorted_models = sorted(self.fitted_models, key=lambda x: x[2])

        # Transform data if needed
        train_transformed = train
        if feature_generator is not None:
            target_col = self.data_config.target[0] if isinstance(
                self.data_config.target, list
            ) else self.data_config.target
            y_train = train[target_col]
            feature_cols = [c for c in train.columns if c != target_col]
            X_train = train[feature_cols]
            X_train_transformed = feature_generator.fit_transform(X_train, y_train)
            if isinstance(X_train_transformed, np.ndarray):
                X_train_transformed = pd.DataFrame(
                    X_train_transformed, index=X_train.index
                )
            train_transformed = pd.concat([X_train_transformed, y_train], axis=1)

        # Refit top-k models
        refitted = []
        for name, model, score in sorted_models[:top_k]:
            if self.verbose:
                logger.info(f"Refitting {name} on full training data...")
            model.fit(train_transformed)
            refitted.append((name, model, score))

        self.fitted_models = refitted
        # Recalculate weights for refitted models
        if refitted:
            scores = np.array([s for _, _, s in refitted])
            self.weights = self._softmax(-scores)

        return self
