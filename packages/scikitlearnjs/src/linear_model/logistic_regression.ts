/**
 * LogisticRegression - Logistic Regression classifier.
 *
 * In the multiclass case, the training algorithm uses a one-vs-rest (OvR)
 * scheme. This class implements regularized logistic regression.
 *
 * Matches sklearn.linear_model.LogisticRegression API 1:1.
 */

import type { NDArray, Backend } from 'numpyjs';
import type { LogisticRegressionOptions } from './types.js';

/**
 * Default options matching sklearn defaults
 */
const DEFAULT_OPTIONS: Required<LogisticRegressionOptions> = {
  C: 1.0,
  fitIntercept: true,
  maxIter: 100,
  tol: 1e-4,
  penalty: 'l2',
  solver: 'lbfgs',
  randomState: undefined as unknown as number,
  multiClass: 'auto',
  verbose: 0,
  warmStart: false,
  classWeight: undefined as unknown as 'balanced' | Map<number, number>,
};

/**
 * Logistic Regression (aka logit, MaxEnt) classifier.
 *
 * @example
 * ```typescript
 * import { createBackend } from 'numpyjs';
 * import { linear_model } from 'scikitlearnjs';
 *
 * const backend = await createBackend('wasm');
 * const clf = new linear_model.LogisticRegression(backend, { C: 1.0 });
 *
 * await clf.fit(X_train, y_train);
 * const predictions = await clf.predict(X_test);
 * const accuracy = await clf.score(X_test, y_test);
 * ```
 */
export class LogisticRegression {
  private backend: Backend;
  private options: Required<LogisticRegressionOptions>;

  // Fitted attributes
  private _coef: NDArray | null = null;
  private _intercept: NDArray | null = null;
  private _classes: NDArray | null = null;
  private _nIter: number[] = [];
  private _nFeaturesIn = 0;
  private _isFitted = false;

  constructor(backend: Backend, options: LogisticRegressionOptions = {}) {
    this.backend = backend;
    this.options = { ...DEFAULT_OPTIONS, ...options };
  }

  // ==================== Fitted Attributes ====================

  /** Coefficient of the features in the decision function */
  get coef(): NDArray {
    this._checkFitted();
    return this._coef!;
  }

  /** Intercept (bias) added to the decision function */
  get intercept(): NDArray {
    this._checkFitted();
    return this._intercept!;
  }

  /** Classes labels */
  get classes(): NDArray {
    this._checkFitted();
    return this._classes!;
  }

  /** Actual number of iterations for each class */
  get nIter(): number[] {
    this._checkFitted();
    return this._nIter;
  }

  /** Number of features seen during fit */
  get nFeaturesIn(): number {
    this._checkFitted();
    return this._nFeaturesIn;
  }

  // ==================== sklearn API Methods ====================

  /**
   * Fit the model according to the given training data.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Target values of shape (n_samples,)
   * @returns The fitted estimator
   */
  async fit(X: NDArray, y: NDArray): Promise<this> {
    this._validateInput(X);
    const [nSamples, nFeatures] = X.shape;
    this._nFeaturesIn = nFeatures;

    // Get unique classes
    const classSet = new Set<number>();
    for (const val of y.data) {
      classSet.add(val);
    }
    const classes = Array.from(classSet).sort((a, b) => a - b);
    this._classes = this.backend.array(classes, [classes.length]);

    const nClasses = classes.length;

    // Binary classification
    if (nClasses === 2) {
      const result = await this._fitBinary(X, y, classes);
      this._coef = result.coef;
      this._intercept = result.intercept;
      this._nIter = [result.nIter];
    } else {
      // Multiclass: one-vs-rest
      const coefs: Float64Array[] = [];
      const intercepts: number[] = [];
      this._nIter = [];

      for (const cls of classes) {
        // Create binary labels
        const yBinary = new Float64Array(nSamples);
        for (let i = 0; i < nSamples; i++) {
          yBinary[i] = y.data[i] === cls ? 1 : 0;
        }
        const yArr = this.backend.array(Array.from(yBinary), [nSamples]);

        const result = await this._fitBinary(X, yArr, [0, 1]);
        coefs.push(result.coef.data as Float64Array);
        intercepts.push(result.intercept.data[0]);
        this._nIter.push(result.nIter);
      }

      // Stack coefficients: shape (n_classes, n_features)
      const coefData = new Float64Array(nClasses * nFeatures);
      for (let c = 0; c < nClasses; c++) {
        for (let f = 0; f < nFeatures; f++) {
          coefData[c * nFeatures + f] = coefs[c][f];
        }
      }
      this._coef = this.backend.array(Array.from(coefData), [nClasses, nFeatures]);
      this._intercept = this.backend.array(intercepts, [nClasses]);
    }

    this._isFitted = true;
    return this;
  }

  /**
   * Predict class labels for samples in X.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predicted class label per sample of shape (n_samples,)
   */
  async predict(X: NDArray): Promise<NDArray> {
    this._checkFitted();
    const proba = await this.predictProba(X);
    const nSamples = X.shape[0];
    const nClasses = this._classes!.data.length;

    const predictions = new Float64Array(nSamples);

    for (let i = 0; i < nSamples; i++) {
      let maxIdx = 0;
      let maxVal = -Infinity;
      for (let j = 0; j < nClasses; j++) {
        const val = proba.data[i * nClasses + j];
        if (val > maxVal) {
          maxVal = val;
          maxIdx = j;
        }
      }
      predictions[i] = this._classes!.data[maxIdx];
    }

    return this.backend.array(Array.from(predictions), [nSamples]);
  }

  /**
   * Probability estimates.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Probability of each class of shape (n_samples, n_classes)
   */
  async predictProba(X: NDArray): Promise<NDArray> {
    this._checkFitted();
    this._validateInput(X);

    const nSamples = X.shape[0];
    const nFeatures = X.shape[1];
    const nClasses = this._classes!.data.length;

    if (nFeatures !== this._nFeaturesIn) {
      throw new Error(
        `X has ${nFeatures} features, but LogisticRegression was fitted with ${this._nFeaturesIn} features`
      );
    }

    const proba = new Float64Array(nSamples * nClasses);

    if (nClasses === 2) {
      // Binary: compute sigmoid
      const coef = this._coef!;
      const intercept = this._intercept!.data[0];

      for (let i = 0; i < nSamples; i++) {
        let z = intercept;
        for (let j = 0; j < nFeatures; j++) {
          z += X.data[i * nFeatures + j] * coef.data[j];
        }
        const p1 = 1 / (1 + Math.exp(-z));
        proba[i * 2] = 1 - p1;
        proba[i * 2 + 1] = p1;
      }
    } else {
      // Multiclass: softmax over OvR scores
      for (let i = 0; i < nSamples; i++) {
        let maxScore = -Infinity;
        const scores: number[] = [];

        for (let c = 0; c < nClasses; c++) {
          let z = this._intercept!.data[c];
          for (let j = 0; j < nFeatures; j++) {
            z += X.data[i * nFeatures + j] * this._coef!.data[c * nFeatures + j];
          }
          scores.push(z);
          if (z > maxScore) maxScore = z;
        }

        // Softmax
        let sumExp = 0;
        for (let c = 0; c < nClasses; c++) {
          scores[c] = Math.exp(scores[c] - maxScore);
          sumExp += scores[c];
        }

        for (let c = 0; c < nClasses; c++) {
          proba[i * nClasses + c] = scores[c] / sumExp;
        }
      }
    }

    return this.backend.array(Array.from(proba), [nSamples, nClasses]);
  }

  /**
   * Log of probability estimates.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Log-probability of each class of shape (n_samples, n_classes)
   */
  async predictLogProba(X: NDArray): Promise<NDArray> {
    const proba = await this.predictProba(X);
    const logProba = new Float64Array(proba.data.length);

    for (let i = 0; i < proba.data.length; i++) {
      logProba[i] = Math.log(Math.max(1e-15, proba.data[i]));
    }

    return this.backend.array(Array.from(logProba), proba.shape);
  }

  /**
   * Return mean accuracy on the given test data and labels.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True labels of shape (n_samples,)
   * @returns Mean accuracy
   */
  async score(X: NDArray, y: NDArray): Promise<number> {
    const predictions = await this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.data.length; i++) {
      if (predictions.data[i] === y.data[i]) {
        correct++;
      }
    }
    return correct / y.data.length;
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
  setParams(params: Partial<LogisticRegressionOptions>): this {
    Object.assign(this.options, params);
    return this;
  }

  // ==================== Internal Methods ====================

  private _checkFitted(): void {
    if (!this._isFitted) {
      throw new Error(
        'This LogisticRegression instance is not fitted yet. Call fit() before using this estimator.'
      );
    }
  }

  private _validateInput(X: NDArray): void {
    if (X.shape.length !== 2) {
      throw new Error(`Expected 2D array, got ${X.shape.length}D array instead`);
    }
  }

  /**
   * Fit binary logistic regression using gradient descent
   */
  private async _fitBinary(
    X: NDArray,
    y: NDArray,
    _classes: number[]
  ): Promise<{ coef: NDArray; intercept: NDArray; nIter: number }> {
    const nSamples = X.shape[0];
    const nFeatures = X.shape[1];

    // Initialize weights
    const w = new Float64Array(nFeatures);
    let b = 0;

    const lr = 0.1;
    const C = this.options.C;
    const lambda = 1 / C; // Regularization strength

    let prevLoss = Infinity;
    let nIter = 0;

    for (let iter = 0; iter < this.options.maxIter; iter++) {
      nIter = iter + 1;

      // Compute predictions
      const predictions = new Float64Array(nSamples);
      for (let i = 0; i < nSamples; i++) {
        let z = b;
        for (let j = 0; j < nFeatures; j++) {
          z += X.data[i * nFeatures + j] * w[j];
        }
        predictions[i] = 1 / (1 + Math.exp(-z));
      }

      // Compute loss
      let loss = 0;
      for (let i = 0; i < nSamples; i++) {
        const p = Math.max(1e-15, Math.min(1 - 1e-15, predictions[i]));
        loss -= y.data[i] * Math.log(p) + (1 - y.data[i]) * Math.log(1 - p);
      }
      loss /= nSamples;

      // Add L2 regularization
      if (this.options.penalty === 'l2') {
        let wNorm = 0;
        for (const wi of w) wNorm += wi * wi;
        loss += (lambda / 2) * wNorm;
      }

      // Check convergence
      if (Math.abs(prevLoss - loss) < this.options.tol) {
        break;
      }
      prevLoss = loss;

      // Compute gradients
      const gradW = new Float64Array(nFeatures);
      let gradB = 0;

      for (let i = 0; i < nSamples; i++) {
        const error = predictions[i] - y.data[i];
        gradB += error;
        for (let j = 0; j < nFeatures; j++) {
          gradW[j] += error * X.data[i * nFeatures + j];
        }
      }

      for (let j = 0; j < nFeatures; j++) {
        gradW[j] /= nSamples;
        // Add L2 regularization gradient
        if (this.options.penalty === 'l2') {
          gradW[j] += lambda * w[j];
        }
      }
      gradB /= nSamples;

      // Update weights
      for (let j = 0; j < nFeatures; j++) {
        w[j] -= lr * gradW[j];
      }
      if (this.options.fitIntercept) {
        b -= lr * gradB;
      }
    }

    return {
      coef: this.backend.array(Array.from(w), [nFeatures]),
      intercept: this.backend.array([b], [1]),
      nIter,
    };
  }
}
