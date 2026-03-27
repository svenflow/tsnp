/**
 * Types for sklearn.neural_network module
 *
 * Matches sklearn.neural_network API 1:1
 */

import type { NDArray } from 'numpyjs';

/**
 * Activation functions supported by MLP
 * Matches sklearn's activation parameter options
 */
export type Activation = 'identity' | 'logistic' | 'tanh' | 'relu';

/**
 * Solvers for weight optimization
 * Matches sklearn's solver parameter options
 */
export type Solver = 'lbfgs' | 'sgd' | 'adam';

/**
 * Learning rate schedule
 * Matches sklearn's learning_rate parameter options
 */
export type LearningRateSchedule = 'constant' | 'invscaling' | 'adaptive';

/**
 * Configuration options for MLPClassifier
 * Matches sklearn.neural_network.MLPClassifier constructor parameters
 */
export interface MLPClassifierOptions {
  /** Sizes of hidden layers. Default: [100] */
  hiddenLayerSizes?: number[];
  /** Activation function. Default: 'relu' */
  activation?: Activation;
  /** Solver for weight optimization. Default: 'adam' */
  solver?: Solver;
  /** L2 penalty (regularization term). Default: 0.0001 */
  alpha?: number;
  /** Size of minibatches. 'auto' means min(200, n_samples). Default: 'auto' */
  batchSize?: number | 'auto';
  /** Learning rate schedule. Default: 'constant' */
  learningRate?: LearningRateSchedule;
  /** Initial learning rate. Default: 0.001 */
  learningRateInit?: number;
  /** Exponent for inverse scaling. Default: 0.5 */
  powerT?: number;
  /** Maximum iterations. Default: 200 */
  maxIter?: number;
  /** Whether to shuffle samples. Default: true */
  shuffle?: boolean;
  /** Random state seed. Default: undefined (random) */
  randomState?: number;
  /** Tolerance for optimization. Default: 1e-4 */
  tol?: number;
  /** Verbose output. Default: false */
  verbose?: boolean;
  /** Reuse previous fit's solution. Default: false */
  warmStart?: boolean;
  /** Momentum for gradient descent. Default: 0.9 */
  momentum?: number;
  /** Use Nesterov's momentum. Default: true */
  nesterovsMomentum?: boolean;
  /** Use early stopping. Default: false */
  earlyStopping?: boolean;
  /** Fraction for validation. Default: 0.1 */
  validationFraction?: number;
  /** Exponential decay rate for first moment. Default: 0.9 */
  beta1?: number;
  /** Exponential decay rate for second moment. Default: 0.999 */
  beta2?: number;
  /** Value for numerical stability. Default: 1e-8 */
  epsilon?: number;
  /** Max epochs without improvement. Default: 10 */
  nIterNoChange?: number;
  /** Max function calls (lbfgs only). Default: 15000 */
  maxFun?: number;
}

/**
 * Configuration options for MLPRegressor
 * Same as MLPClassifier but for regression
 */
export interface MLPRegressorOptions extends MLPClassifierOptions {}

/**
 * Fitted attributes available after training
 */
export interface MLPFittedAttributes {
  /** Weight matrices for each layer */
  coefs: NDArray[];
  /** Bias vectors for each layer */
  intercepts: NDArray[];
  /** Class labels (classifier only) */
  classes?: NDArray;
  /** Number of iterations run */
  nIter: number;
  /** Number of output neurons */
  nOutputs: number;
  /** Number of layers */
  nLayers: number;
  /** Names of output neurons */
  outActivation: Activation | 'softmax';
  /** Current loss value */
  loss: number;
  /** Best loss value during training */
  bestLoss: number;
  /** Loss curve (loss values per iteration) */
  lossCurve: number[];
  /** Validation scores (if early_stopping) */
  validationScores?: number[];
}

/**
 * Base interface for estimators following sklearn's API
 */
export interface BaseEstimator {
  /** Get parameters for this estimator */
  getParams(deep?: boolean): Record<string, unknown>;
  /** Set parameters for this estimator */
  setParams(params: Record<string, unknown>): this;
}

/**
 * Interface for classifiers
 */
export interface ClassifierMixin {
  /** Fit the model and return self */
  fit(X: NDArray, y: NDArray): Promise<this>;
  /** Predict class labels */
  predict(X: NDArray): Promise<NDArray>;
  /** Predict class probabilities */
  predictProba(X: NDArray): Promise<NDArray>;
  /** Return mean accuracy on given test data */
  score(X: NDArray, y: NDArray): Promise<number>;
}

/**
 * Interface for regressors
 */
export interface RegressorMixin {
  /** Fit the model and return self */
  fit(X: NDArray, y: NDArray): Promise<this>;
  /** Predict target values */
  predict(X: NDArray): Promise<NDArray>;
  /** Return R² score */
  score(X: NDArray, y: NDArray): Promise<number>;
}
