/**
 * StandardScaler - Standardize features by removing the mean and scaling to unit variance
 *
 * The standard score of a sample x is calculated as:
 *   z = (x - u) / s
 * where u is the mean of the training samples and s is the standard deviation.
 *
 * Matches sklearn.preprocessing.StandardScaler API 1:1.
 */

import type { NDArray, Backend } from 'numpyjs';
import type { StandardScalerOptions, TransformerMixin } from './types.js';

/**
 * Default options matching sklearn defaults
 */
const DEFAULT_OPTIONS: Required<StandardScalerOptions> = {
  withMean: true,
  withStd: true,
};

/**
 * Standardize features by removing the mean and scaling to unit variance.
 *
 * @example
 * ```typescript
 * import { createBackend } from 'numpyjs';
 * import { preprocessing } from 'scikitlearnjs';
 *
 * const backend = await createBackend('wasm');
 * const scaler = new preprocessing.StandardScaler(backend);
 *
 * const X_train_scaled = await scaler.fitTransform(X_train);
 * const X_test_scaled = await scaler.transform(X_test);
 * ```
 */
export class StandardScaler implements TransformerMixin {
  private backend: Backend;
  private options: Required<StandardScalerOptions>;

  // Fitted attributes
  private _mean: NDArray | null = null;
  private _var: NDArray | null = null;
  private _scale: NDArray | null = null;
  private _nFeaturesIn = 0;
  private _nSamplesSeen = 0;
  private _isFitted = false;

  constructor(backend: Backend, options: StandardScalerOptions = {}) {
    this.backend = backend;
    this.options = { ...DEFAULT_OPTIONS, ...options };
  }

  // ==================== Fitted Attributes ====================

  /** Per-feature mean */
  get mean(): NDArray {
    this._checkFitted();
    return this._mean!;
  }

  /** Per-feature variance */
  get var(): NDArray {
    this._checkFitted();
    return this._var!;
  }

  /** Per-feature standard deviation */
  get scale(): NDArray {
    this._checkFitted();
    return this._scale!;
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
   * Compute the mean and std to be used for later scaling.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @returns The fitted scaler
   */
  async fit(X: NDArray): Promise<this> {
    this._validateInput(X);

    const [nSamples, nFeatures] = X.shape;
    this._nSamplesSeen = nSamples;
    this._nFeaturesIn = nFeatures;

    // Compute mean for each feature
    const meanData = new Float64Array(nFeatures);
    for (let j = 0; j < nFeatures; j++) {
      let sum = 0;
      for (let i = 0; i < nSamples; i++) {
        sum += X.data[i * nFeatures + j];
      }
      meanData[j] = sum / nSamples;
    }
    this._mean = this.backend.array(Array.from(meanData), [nFeatures]);

    // Compute variance for each feature
    const varData = new Float64Array(nFeatures);
    for (let j = 0; j < nFeatures; j++) {
      let sum = 0;
      for (let i = 0; i < nSamples; i++) {
        const diff = X.data[i * nFeatures + j]! - meanData[j]!;
        sum += diff * diff;
      }
      varData[j] = sum / nSamples;
    }
    this._var = this.backend.array(Array.from(varData), [nFeatures]);

    // Compute scale (std) - handle zero variance
    const scaleData = new Float64Array(nFeatures);
    for (let j = 0; j < nFeatures; j++) {
      const std = Math.sqrt(varData[j]!);
      // Like sklearn, set scale to 1 for constant features
      scaleData[j] = std === 0 ? 1 : std;
    }
    this._scale = this.backend.array(Array.from(scaleData), [nFeatures]);

    this._isFitted = true;
    return this;
  }

  /**
   * Perform standardization by centering and scaling.
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
        `X has ${nFeatures} features, but StandardScaler was fitted with ${this._nFeaturesIn} features`
      );
    }

    const result = new Float64Array(X.data.length);

    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        let val = X.data[i * nFeatures + j];

        // Center
        if (this.options.withMean) {
          val -= this._mean!.data[j];
        }

        // Scale
        if (this.options.withStd) {
          val /= this._scale!.data[j];
        }

        result[i * nFeatures + j] = val;
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
   * Scale back the data to the original representation.
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
        `X has ${nFeatures} features, but StandardScaler was fitted with ${this._nFeaturesIn} features`
      );
    }

    const result = new Float64Array(X.data.length);

    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        let val = X.data[i * nFeatures + j];

        // Unscale
        if (this.options.withStd) {
          val *= this._scale!.data[j];
        }

        // Uncenter
        if (this.options.withMean) {
          val += this._mean!.data[j];
        }

        result[i * nFeatures + j] = val;
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
  setParams(params: Partial<StandardScalerOptions>): this {
    Object.assign(this.options, params);
    return this;
  }

  // ==================== Internal Methods ====================

  private _checkFitted(): void {
    if (!this._isFitted) {
      throw new Error(
        'This StandardScaler instance is not fitted yet. Call fit() before using this transformer.'
      );
    }
  }

  private _validateInput(X: NDArray): void {
    if (X.shape.length !== 2) {
      throw new Error(`Expected 2D array, got ${X.shape.length}D array instead`);
    }
  }
}
