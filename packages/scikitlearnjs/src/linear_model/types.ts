/**
 * Types for sklearn.linear_model module
 *
 * Matches sklearn.linear_model API 1:1
 */

import type { NDArray } from 'numpyjs';

/**
 * Options for LinearRegression
 */
export interface LinearRegressionOptions {
  /** Whether to calculate intercept. Default: true */
  fitIntercept?: boolean;
  /** If True, copy X; if False, may overwrite. Default: true */
  copy?: boolean;
  /** Number of jobs for computation (-1 for all). Default: 1 */
  nJobs?: number;
  /** This parameter is ignored in JavaScript version */
  positive?: boolean;
}

/**
 * Options for LogisticRegression
 */
export interface LogisticRegressionOptions {
  /** Regularization strength (inverse). Default: 1.0 */
  C?: number;
  /** Whether to calculate intercept. Default: true */
  fitIntercept?: boolean;
  /** Maximum iterations. Default: 100 */
  maxIter?: number;
  /** Tolerance for stopping. Default: 1e-4 */
  tol?: number;
  /** Regularization penalty ('l1', 'l2', 'elasticnet'). Default: 'l2' */
  penalty?: 'l1' | 'l2' | 'elasticnet' | 'none';
  /** Solver algorithm. Default: 'lbfgs' */
  solver?: 'newton-cg' | 'lbfgs' | 'liblinear' | 'sag' | 'saga';
  /** Random state for shuffling. Default: undefined */
  randomState?: number;
  /** Multiclass strategy. Default: 'auto' */
  multiClass?: 'auto' | 'ovr' | 'multinomial';
  /** Verbose output. Default: 0 */
  verbose?: number;
  /** Warm start from previous solution. Default: false */
  warmStart?: boolean;
  /** Class weights. Default: undefined */
  classWeight?: 'balanced' | Map<number, number>;
}

/**
 * Fitted attributes for LinearRegression
 */
export interface LinearRegressionAttributes {
  /** Estimated coefficients */
  coef: NDArray;
  /** Independent term (intercept) */
  intercept: number | NDArray;
  /** Number of features */
  nFeaturesIn: number;
  /** Rank of X */
  rank?: number;
  /** Singular values of X */
  singular?: NDArray;
}

/**
 * Fitted attributes for LogisticRegression
 */
export interface LogisticRegressionAttributes {
  /** Coefficients (weights) */
  coef: NDArray;
  /** Intercept (bias) */
  intercept: NDArray;
  /** Class labels */
  classes: NDArray;
  /** Number of iterations */
  nIter: number[];
  /** Number of features */
  nFeaturesIn: number;
}
