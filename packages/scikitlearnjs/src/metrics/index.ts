/**
 * sklearn.metrics module
 *
 * Model evaluation metrics for classification and regression.
 * Matches sklearn.metrics API 1:1.
 */

// Classification metrics
export {
  accuracyScore,
  precisionScore,
  recallScore,
  f1Score,
  confusionMatrix,
  logLoss,
} from './classification.js';

// Regression metrics
export {
  meanSquaredError,
  rootMeanSquaredError,
  meanAbsoluteError,
  r2Score,
  meanAbsolutePercentageError,
  explainedVarianceScore,
} from './regression.js';

// Types
export type {
  AverageType,
  ClassificationMetricOptions,
  RegressionMetricOptions,
  ConfusionMatrixResult,
} from './types.js';
