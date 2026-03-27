/**
 * Types for sklearn.metrics module
 *
 * Matches sklearn.metrics API 1:1
 */

import type { NDArray } from 'numpyjs';

/**
 * Averaging strategies for multi-class/multi-label metrics
 */
export type AverageType = 'micro' | 'macro' | 'weighted' | 'samples' | null;

/**
 * Options for classification metrics
 */
export interface ClassificationMetricOptions {
  /** Labels to include */
  labels?: number[];
  /** Averaging strategy */
  average?: AverageType;
  /** Sample weights */
  sampleWeight?: NDArray;
  /** Value to return when there's a zero division */
  zeroDevisionValue?: number;
}

/**
 * Options for regression metrics
 */
export interface RegressionMetricOptions {
  /** Sample weights */
  sampleWeight?: NDArray;
  /** Whether to compute multi-output scores */
  multioutput?: 'raw_values' | 'uniform_average' | 'variance_weighted';
}

/**
 * Confusion matrix result
 */
export interface ConfusionMatrixResult {
  /** The confusion matrix as NDArray */
  matrix: NDArray;
  /** Class labels */
  labels: number[];
}
