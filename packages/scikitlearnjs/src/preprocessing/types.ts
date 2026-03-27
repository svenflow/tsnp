/**
 * Types for sklearn.preprocessing module
 *
 * Matches sklearn.preprocessing API 1:1
 */

import type { NDArray } from 'numpyjs';

/**
 * Interface for transformers
 */
export interface TransformerMixin {
  /** Fit the transformer and return self */
  fit(X: NDArray): Promise<this>;
  /** Transform the data */
  transform(X: NDArray): Promise<NDArray>;
  /** Fit and transform in one step */
  fitTransform(X: NDArray): Promise<NDArray>;
}

/**
 * Configuration options for StandardScaler
 */
export interface StandardScalerOptions {
  /** Whether to center the data. Default: true */
  withMean?: boolean;
  /** Whether to scale to unit variance. Default: true */
  withStd?: boolean;
}

/**
 * Configuration options for MinMaxScaler
 */
export interface MinMaxScalerOptions {
  /** Desired range of transformed data. Default: [0, 1] */
  featureRange?: [number, number];
}

/**
 * Fitted attributes for StandardScaler
 */
export interface StandardScalerAttributes {
  /** Per-feature mean */
  mean: NDArray;
  /** Per-feature variance */
  var: NDArray;
  /** Per-feature standard deviation */
  scale: NDArray;
  /** Number of features seen during fit */
  nFeaturesIn: number;
  /** Number of samples seen during fit */
  nSamplesSeen: number;
}

/**
 * Fitted attributes for MinMaxScaler
 */
export interface MinMaxScalerAttributes {
  /** Per-feature minimum */
  dataMin: NDArray;
  /** Per-feature maximum */
  dataMax: NDArray;
  /** Per-feature range (max - min) */
  dataRange: NDArray;
  /** Per-feature scaling factor */
  scale: NDArray;
  /** Per-feature offset */
  min: NDArray;
  /** Number of features seen during fit */
  nFeaturesIn: number;
  /** Number of samples seen during fit */
  nSamplesSeen: number;
}
