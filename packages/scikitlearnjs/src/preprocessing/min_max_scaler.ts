/**
 * MinMaxScaler - Transform features by scaling each feature to a given range
 *
 * This estimator scales and translates each feature individually such that
 * it is in the given range on the training set, e.g. between zero and one.
 *
 * The transformation is given by:
 *   X_std = (X - X.min) / (X.max - X.min)
 *   X_scaled = X_std * (max - min) + min
 *
 * where min, max = feature_range.
 *
 * Matches sklearn.preprocessing.MinMaxScaler API 1:1.
 */

import type { NDArray, Backend } from 'numpyjs';
import type { MinMaxScalerOptions, TransformerMixin } from './types.js';

/**
 * Default options matching sklearn defaults
 */
const DEFAULT_OPTIONS: Required<MinMaxScalerOptions> = {
  featureRange: [0, 1],
};

/**
 * Transform features by scaling each feature to a given range.
 *
 * @example
 * ```typescript
 * import { createBackend } from 'numpyjs';
 * import { preprocessing } from 'scikitlearnjs';
 *
 * const backend = await createBackend('wasm');
 * const scaler = new preprocessing.MinMaxScaler(backend, { featureRange: [0, 1] });
 *
 * const X_train_scaled = await scaler.fitTransform(X_train);
 * const X_test_scaled = await scaler.transform(X_test);
 * ```
 */
export class MinMaxScaler implements TransformerMixin {
  private backend: Backend;
  private options: Required<MinMaxScalerOptions>;

  // Fitted attributes
  private _dataMin: NDArray | null = null;
  private _dataMax: NDArray | null = null;
  private _dataRange: NDArray | null = null;
  private _scale: NDArray | null = null;
  private _min: NDArray | null = null;
  private _nFeaturesIn = 0;
  private _nSamplesSeen = 0;
  private _isFitted = false;

  constructor(backend: Backend, options: MinMaxScalerOptions = {}) {
    this.backend = backend;
    this.options = { ...DEFAULT_OPTIONS, ...options };
  }

  // ==================== Fitted Attributes ====================

  /** Per-feature minimum seen in the data */
  get dataMin(): NDArray {
    this._checkFitted();
    return this._dataMin!;
  }

  /** Per-feature maximum seen in the data */
  get dataMax(): NDArray {
    this._checkFitted();
    return this._dataMax!;
  }

  /** Per-feature range (data_max - data_min) */
  get dataRange(): NDArray {
    this._checkFitted();
    return this._dataRange!;
  }

  /** Per-feature scaling factor to apply */
  get scale(): NDArray {
    this._checkFitted();
    return this._scale!;
  }

  /** Per-feature adjustment for minimum */
  get min(): NDArray {
    this._checkFitted();
    return this._min!;
  }

  /** Number of features seen during fit */
  get nFeaturesIn(): number {
    this._checkFitted();
    return this._nFeaturesIn;
  }

  /** Number of samples seen during fit */
  get nSamplesSeen(): number {
    this._checkFitted();
    return this._nSamplesSeen;
  }

  // ==================== sklearn API Methods ====================

  /**
   * Compute the minimum and maximum to be used for later scaling.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @returns The fitted scaler
   */
  async fit(X: NDArray): Promise<this> {
    this._validateInput(X);

    const [nSamples, nFeatures] = X.shape;
    this._nSamplesSeen = nSamples;
    this._nFeaturesIn = nFeatures;

    const [featureMin, featureMax] = this.options.featureRange;
    const featureRangeSize = featureMax - featureMin;

    // Compute min and max for each feature
    const minData = new Float64Array(nFeatures);
    const maxData = new Float64Array(nFeatures);

    // Initialize with first row
    for (let j = 0; j < nFeatures; j++) {
      minData[j] = X.data[j]!;
      maxData[j] = X.data[j]!;
    }

    // Find min/max
    for (let i = 1; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        const val = X.data[i * nFeatures + j]!;
        if (val < minData[j]!) minData[j] = val;
        if (val > maxData[j]!) maxData[j] = val;
      }
    }

    this._dataMin = this.backend.array(Array.from(minData), [nFeatures]);
    this._dataMax = this.backend.array(Array.from(maxData), [nFeatures]);

    // Compute range
    const rangeData = new Float64Array(nFeatures);
    for (let j = 0; j < nFeatures; j++) {
      rangeData[j] = maxData[j]! - minData[j]!;
    }
    this._dataRange = this.backend.array(Array.from(rangeData), [nFeatures]);

    // Compute scale: (feature_max - feature_min) / (data_max - data_min)
    const scaleData = new Float64Array(nFeatures);
    for (let j = 0; j < nFeatures; j++) {
      // Handle zero range (constant feature)
      scaleData[j] = rangeData[j]! === 0 ? 0 : featureRangeSize / rangeData[j]!;
    }
    this._scale = this.backend.array(Array.from(scaleData), [nFeatures]);

    // Compute min adjustment: feature_min - data_min * scale
    const minAdjData = new Float64Array(nFeatures);
    for (let j = 0; j < nFeatures; j++) {
      minAdjData[j] = featureMin - minData[j]! * scaleData[j]!;
    }
    this._min = this.backend.array(Array.from(minAdjData), [nFeatures]);

    this._isFitted = true;
    return this;
  }

  /**
   * Scale features of X according to feature_range.
   *
   * @param X - Data to transform of shape (n_samples, n_features)
   * @returns Transformed data
   */
  async transform(X: NDArray): Promise<NDArray> {
    this._checkFitted();
    this._validateInput(X);

    const [nSamples, nFeatures] = X.shape;

    if (nFeatures !== this._nFeaturesIn) {
      throw new Error(
        `X has ${nFeatures} features, but MinMaxScaler was fitted with ${this._nFeaturesIn} features`
      );
    }

    const result = new Float64Array(X.data.length);

    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        // X_scaled = X * scale + min
        result[i * nFeatures + j] =
          X.data[i * nFeatures + j] * this._scale!.data[j] + this._min!.data[j];
      }
    }

    return this.backend.array(Array.from(result), [nSamples, nFeatures]);
  }

  /**
   * Fit to data, then transform it.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @returns Transformed data
   */
  async fitTransform(X: NDArray): Promise<NDArray> {
    await this.fit(X);
    return this.transform(X);
  }

  /**
   * Undo the scaling of X according to feature_range.
   *
   * @param X - Scaled data of shape (n_samples, n_features)
   * @returns Original-scale data
   */
  async inverseTransform(X: NDArray): Promise<NDArray> {
    this._checkFitted();
    this._validateInput(X);

    const [nSamples, nFeatures] = X.shape;

    if (nFeatures !== this._nFeaturesIn) {
      throw new Error(
        `X has ${nFeatures} features, but MinMaxScaler was fitted with ${this._nFeaturesIn} features`
      );
    }

    const result = new Float64Array(X.data.length);

    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        // X = (X_scaled - min) / scale
        const scale = this._scale!.data[j];
        if (scale === 0) {
          // Constant feature, return data_min
          result[i * nFeatures + j] = this._dataMin!.data[j];
        } else {
          result[i * nFeatures + j] =
            (X.data[i * nFeatures + j] - this._min!.data[j]) / scale;
        }
      }
    }

    return this.backend.array(Array.from(result), [nSamples, nFeatures]);
  }

  /**
   * Get parameters for this estimator.
   */
  getParams(_deep = true): Record<string, unknown> {
    return { ...this.options };
  }

  /**
   * Set parameters for this estimator.
   */
  setParams(params: Partial<MinMaxScalerOptions>): this {
    Object.assign(this.options, params);
    return this;
  }

  // ==================== Internal Methods ====================

  private _checkFitted(): void {
    if (!this._isFitted) {
      throw new Error(
        'This MinMaxScaler instance is not fitted yet. Call fit() before using this transformer.'
      );
    }
  }

  private _validateInput(X: NDArray): void {
    if (X.shape.length !== 2) {
      throw new Error(`Expected 2D array, got ${X.shape.length}D array instead`);
    }
  }
}
