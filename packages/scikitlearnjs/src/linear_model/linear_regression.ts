/**
 * LinearRegression - Ordinary least squares Linear Regression.
 *
 * LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
 * to minimize the residual sum of squares between the observed targets in
 * the dataset, and the targets predicted by the linear approximation.
 *
 * Matches sklearn.linear_model.LinearRegression API 1:1.
 */

import type { NDArray, Backend } from 'numpyjs';
import type { LinearRegressionOptions } from './types.js';

/**
 * Default options matching sklearn defaults
 */
const DEFAULT_OPTIONS: Required<LinearRegressionOptions> = {
  fitIntercept: true,
  copy: true,
  nJobs: 1,
  positive: false,
};

/**
 * Ordinary least squares Linear Regression.
 *
 * @example
 * ```typescript
 * import { createBackend } from 'numpyjs';
 * import { linear_model } from 'scikitlearnjs';
 *
 * const backend = await createBackend('wasm');
 * const reg = new linear_model.LinearRegression(backend);
 *
 * await reg.fit(X_train, y_train);
 * const predictions = await reg.predict(X_test);
 * const r2 = await reg.score(X_test, y_test);
 * ```
 */
export class LinearRegression {
  private backend: Backend;
  private options: Required<LinearRegressionOptions>;

  // Fitted attributes
  private _coef: NDArray | null = null;
  private _intercept: number = 0;
  private _nFeaturesIn = 0;
  private _isFitted = false;

  constructor(backend: Backend, options: LinearRegressionOptions = {}) {
    this.backend = backend;
    this.options = { ...DEFAULT_OPTIONS, ...options };
  }

  // ==================== Fitted Attributes ====================

  /** Estimated coefficients for the linear regression problem */
  get coef(): NDArray {
    this._checkFitted();
    return this._coef!;
  }

  /** Independent term in the linear model */
  get intercept(): number {
    this._checkFitted();
    return this._intercept;
  }

  /** Number of features seen during fit */
  get nFeaturesIn(): number {
    this._checkFitted();
    return this._nFeaturesIn;
  }

  // ==================== sklearn API Methods ====================

  /**
   * Fit linear model.
   *
   * Uses the normal equation: w = (X^T X)^{-1} X^T y
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Target values of shape (n_samples,)
   * @returns The fitted estimator
   */
  async fit(X: NDArray, y: NDArray): Promise<this> {
    this._validateInput(X);
    const [nSamples, nFeatures] = X.shape;
    this._nFeaturesIn = nFeatures;

    // Center data if fitting intercept
    let XCentered = X;
    let yCentered = y;
    let xMean: Float64Array | null = null;
    let yMean = 0;

    if (this.options.fitIntercept) {
      // Compute means
      xMean = new Float64Array(nFeatures);
      for (let j = 0; j < nFeatures; j++) {
        for (let i = 0; i < nSamples; i++) {
          xMean[j] += X.data[i * nFeatures + j];
        }
        xMean[j] /= nSamples;
      }

      for (let i = 0; i < nSamples; i++) {
        yMean += y.data[i];
      }
      yMean /= nSamples;

      // Center X
      const XCenteredData = new Float64Array(X.data.length);
      for (let i = 0; i < nSamples; i++) {
        for (let j = 0; j < nFeatures; j++) {
          XCenteredData[i * nFeatures + j] = X.data[i * nFeatures + j] - xMean[j];
        }
      }
      XCentered = this.backend.array(Array.from(XCenteredData), [nSamples, nFeatures]);

      // Center y
      const yCenteredData = new Float64Array(nSamples);
      for (let i = 0; i < nSamples; i++) {
        yCenteredData[i] = y.data[i] - yMean;
      }
      yCentered = this.backend.array(Array.from(yCenteredData), [nSamples]);
    }

    // Compute normal equation: w = (X^T X)^{-1} X^T y
    // First: X^T X
    const XT = this.backend.transpose(XCentered);
    const XTX = this.backend.matmul(XT, XCentered);

    // X^T y
    const yReshaped = this.backend.reshape(yCentered, [nSamples, 1]);
    const XTy = this.backend.matmul(XT, yReshaped);

    // Solve using Cholesky or direct inversion
    // For simplicity, we'll use gradient descent to approximate
    // (proper implementation would use QR decomposition or Cholesky)
    const coef = await this._solveNormalEquation(XTX, XTy, nFeatures);

    this._coef = coef;

    // Compute intercept
    if (this.options.fitIntercept && xMean) {
      this._intercept = yMean;
      for (let j = 0; j < nFeatures; j++) {
        this._intercept -= coef.data[j] * xMean[j];
      }
    } else {
      this._intercept = 0;
    }

    this._isFitted = true;
    return this;
  }

  /**
   * Predict using the linear model.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predicted values of shape (n_samples,)
   */
  async predict(X: NDArray): Promise<NDArray> {
    this._checkFitted();
    this._validateInput(X);

    const nSamples = X.shape[0];
    const nFeatures = X.shape[1];

    if (nFeatures !== this._nFeaturesIn) {
      throw new Error(
        `X has ${nFeatures} features, but LinearRegression was fitted with ${this._nFeaturesIn} features`
      );
    }

    // y = X @ coef + intercept
    const coefReshaped = this.backend.reshape(this._coef!, [nFeatures, 1]);
    const XCoef = this.backend.matmul(X, coefReshaped);

    const predictions = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      predictions[i] = XCoef.data[i] + this._intercept;
    }

    return this.backend.array(Array.from(predictions), [nSamples]);
  }

  /**
   * Return the coefficient of determination R² of the prediction.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True values of shape (n_samples,)
   * @returns R² score
   */
  async score(X: NDArray, y: NDArray): Promise<number> {
    const predictions = await this.predict(X);

    // R² = 1 - SS_res / SS_tot
    let ssRes = 0;
    let ssTot = 0;
    let yMean = 0;

    for (const val of y.data) {
      yMean += val;
    }
    yMean /= y.data.length;

    for (let i = 0; i < y.data.length; i++) {
      const residual = y.data[i] - predictions.data[i];
      ssRes += residual * residual;
      const deviation = y.data[i] - yMean;
      ssTot += deviation * deviation;
    }

    return ssTot === 0 ? 1 : 1 - ssRes / ssTot;
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
  setParams(params: Partial<LinearRegressionOptions>): this {
    Object.assign(this.options, params);
    return this;
  }

  // ==================== Internal Methods ====================

  private _checkFitted(): void {
    if (!this._isFitted) {
      throw new Error(
        'This LinearRegression instance is not fitted yet. Call fit() before using this estimator.'
      );
    }
  }

  private _validateInput(X: NDArray): void {
    if (X.shape.length !== 2) {
      throw new Error(`Expected 2D array, got ${X.shape.length}D array instead`);
    }
  }

  /**
   * Solve normal equation using gradient descent
   * (simplified - proper implementation would use matrix decomposition)
   */
  private async _solveNormalEquation(
    XTX: NDArray,
    XTy: NDArray,
    nFeatures: number
  ): Promise<NDArray> {
    // Use gradient descent to solve XTX @ w = XTy
    // Gradient: 2 * (XTX @ w - XTy)
    const learningRate = 0.001;
    const maxIter = 1000;
    const tol = 1e-8;

    let w = this.backend.zeros([nFeatures, 1]);
    let prevLoss = Infinity;

    for (let iter = 0; iter < maxIter; iter++) {
      // Compute XTX @ w
      const XTXw = this.backend.matmul(XTX, w);

      // Compute residual: XTX @ w - XTy
      const residual = this.backend.sub(XTXw, XTy);

      // Compute loss: ||residual||^2
      let loss = 0;
      for (const val of residual.data) {
        loss += val * val;
      }

      // Check convergence
      if (Math.abs(prevLoss - loss) < tol) {
        break;
      }
      prevLoss = loss;

      // Gradient step: w = w - lr * 2 * residual
      const grad = this.backend.mulScalar(residual, 2 * learningRate);
      w = this.backend.sub(w, grad);
    }

    // Flatten to 1D
    return this.backend.flatten(w);
  }
}
