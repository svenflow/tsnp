/**
 * scikitlearnjs - scikit-learn for JavaScript
 *
 * Machine learning library built on top of numpyjs.
 * Matches sklearn Python API 1:1.
 *
 * @example
 * ```typescript
 * import { createBackend } from 'numpyjs';
 * import { neural_network, preprocessing, metrics } from 'scikitlearnjs';
 *
 * const backend = await createBackend('wasm');
 *
 * // Scale features
 * const scaler = new preprocessing.StandardScaler(backend);
 * const X_scaled = await scaler.fitTransform(X_train);
 *
 * // Train classifier
 * const clf = new neural_network.MLPClassifier(backend, {
 *   hiddenLayerSizes: [100, 50],
 *   activation: 'relu',
 * });
 * await clf.fit(X_scaled, y_train);
 *
 * // Predict and evaluate
 * const predictions = await clf.predict(X_test);
 * const accuracy = metrics.accuracyScore(y_test, predictions);
 * ```
 */

// Module exports (sklearn-style namespace access)
export * as neural_network from './neural_network/index.js';
export * as preprocessing from './preprocessing/index.js';
export * as metrics from './metrics/index.js';
export * as model_selection from './model_selection/index.js';
export * as linear_model from './linear_model/index.js';
export * as cluster from './cluster/index.js';

// Re-export individual classes for convenience
export { MLPClassifier, MLPRegressor } from './neural_network/index.js';
export { StandardScaler, MinMaxScaler } from './preprocessing/index.js';
export { LinearRegression, LogisticRegression } from './linear_model/index.js';
export { KMeans } from './cluster/index.js';

// Re-export commonly used functions
export {
  accuracyScore,
  precisionScore,
  recallScore,
  f1Score,
  confusionMatrix,
  meanSquaredError,
  meanAbsoluteError,
  r2Score,
} from './metrics/index.js';

export { trainTestSplit } from './model_selection/index.js';

// Export types - neural_network
export type {
  MLPClassifierOptions,
  MLPRegressorOptions,
  Activation,
  Solver,
  LearningRateSchedule,
  MLPFittedAttributes,
  BaseEstimator,
  ClassifierMixin,
  RegressorMixin,
} from './neural_network/index.js';

// Export types - preprocessing
export type {
  StandardScalerOptions,
  MinMaxScalerOptions,
  StandardScalerAttributes,
  MinMaxScalerAttributes,
  TransformerMixin,
} from './preprocessing/index.js';

// Export types - metrics
export type {
  AverageType,
  ClassificationMetricOptions,
  RegressionMetricOptions,
} from './metrics/index.js';

// Export types - model_selection
export type {
  TrainTestSplitOptions,
  TrainTestSplitResult,
} from './model_selection/index.js';

// Export types - linear_model
export type {
  LinearRegressionOptions,
  LogisticRegressionOptions,
} from './linear_model/index.js';

// Export types - cluster
export type {
  KMeansOptions,
  InitMethod,
} from './cluster/index.js';

export const version = '0.1.0';
