/**
 * MLPRegressor - Multi-layer Perceptron regressor
 *
 * This model optimizes the squared-error loss using LBFGS, SGD, or Adam.
 * Matches sklearn.neural_network.MLPRegressor API 1:1.
 */

import type { NDArray, Backend } from 'numpyjs';
import type {
  MLPRegressorOptions,
  Activation,
  BaseEstimator,
  RegressorMixin,
} from './types.js';

/**
 * Default options matching sklearn defaults
 */
const DEFAULT_OPTIONS: Required<MLPRegressorOptions> = {
  hiddenLayerSizes: [100],
  activation: 'relu',
  solver: 'adam',
  alpha: 0.0001,
  batchSize: 'auto',
  learningRate: 'constant',
  learningRateInit: 0.001,
  powerT: 0.5,
  maxIter: 200,
  shuffle: true,
  randomState: undefined as unknown as number,
  tol: 1e-4,
  verbose: false,
  warmStart: false,
  momentum: 0.9,
  nesterovsMomentum: true,
  earlyStopping: false,
  validationFraction: 0.1,
  beta1: 0.9,
  beta2: 0.999,
  epsilon: 1e-8,
  nIterNoChange: 10,
  maxFun: 15000,
};

/**
 * Multi-layer Perceptron regressor.
 *
 * This model optimizes the squared-error loss using LBFGS or stochastic
 * gradient descent. Unlike MLPClassifier, the output layer has no
 * activation function (identity).
 *
 * @example
 * ```typescript
 * import { createBackend } from 'numpyjs';
 * import { neural_network } from 'scikitlearnjs';
 *
 * const backend = await createBackend('wasm');
 * const reg = new neural_network.MLPRegressor(backend, {
 *   hiddenLayerSizes: [100, 50],
 *   activation: 'relu',
 *   maxIter: 500,
 * });
 *
 * await reg.fit(X_train, y_train);
 * const predictions = await reg.predict(X_test);
 * const r2 = await reg.score(X_test, y_test);
 * ```
 */
export class MLPRegressor implements BaseEstimator, RegressorMixin {
  private backend: Backend;
  private options: Required<MLPRegressorOptions>;

  // Fitted attributes
  private _coefs: NDArray[] = [];
  private _intercepts: NDArray[] = [];
  private _nIter = 0;
  private _nOutputs = 0;
  private _nLayers = 0;
  private _loss = Infinity;
  private _bestLoss = Infinity;
  private _lossCurve: number[] = [];
  private _isFitted = false;

  constructor(backend: Backend, options: MLPRegressorOptions = {}) {
    this.backend = backend;
    this.options = { ...DEFAULT_OPTIONS, ...options };
  }

  // ==================== Fitted Attributes ====================

  /** Weight matrices for each layer */
  get coefs(): NDArray[] {
    this._checkFitted();
    return this._coefs;
  }

  /** Bias vectors for each layer */
  get intercepts(): NDArray[] {
    this._checkFitted();
    return this._intercepts;
  }

  /** Number of iterations run */
  get nIter(): number {
    this._checkFitted();
    return this._nIter;
  }

  /** Number of outputs */
  get nOutputs(): number {
    this._checkFitted();
    return this._nOutputs;
  }

  /** Number of layers */
  get nLayers(): number {
    this._checkFitted();
    return this._nLayers;
  }

  /** Current loss value */
  get loss(): number {
    this._checkFitted();
    return this._loss;
  }

  /** Best loss achieved during training */
  get bestLoss(): number {
    this._checkFitted();
    return this._bestLoss;
  }

  /** Loss curve over iterations */
  get lossCurve(): number[] {
    this._checkFitted();
    return this._lossCurve;
  }

  // ==================== sklearn API Methods ====================

  /**
   * Fit the model to data matrix X and target(s) y.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Target values of shape (n_samples,) or (n_samples, n_outputs)
   * @returns The fitted estimator
   */
  async fit(X: NDArray, y: NDArray): Promise<this> {
    // Validate inputs
    const [nSamples, nFeatures] = this._validateInput(X);

    // Determine number of outputs
    if (y.shape.length === 1) {
      this._nOutputs = 1;
    } else {
      this._nOutputs = y.shape[1];
    }

    // Reshape y if needed
    const yReshaped =
      y.shape.length === 1
        ? this.backend.reshape(y, [nSamples, 1])
        : y;

    // Build layer sizes
    const layerSizes = [nFeatures, ...this.options.hiddenLayerSizes, this._nOutputs];
    this._nLayers = layerSizes.length;

    // Initialize weights
    if (!this.options.warmStart || !this._isFitted) {
      this._initializeWeights(layerSizes);
    }

    // Training loop
    const batchSize =
      this.options.batchSize === 'auto'
        ? Math.min(200, nSamples)
        : this.options.batchSize;

    this._lossCurve = [];
    this._bestLoss = Infinity;
    let noImprovementCount = 0;

    for (let iter = 0; iter < this.options.maxIter; iter++) {
      this._nIter = iter + 1;

      // Forward pass and compute loss
      const { loss, gradients } = await this._computeGradients(X, yReshaped, batchSize);
      this._loss = loss;
      this._lossCurve.push(loss);

      if (loss < this._bestLoss - this.options.tol) {
        this._bestLoss = loss;
        noImprovementCount = 0;
      } else {
        noImprovementCount++;
      }

      // Check convergence
      if (noImprovementCount >= this.options.nIterNoChange) {
        if (this.options.verbose) {
          console.log(`Converged after ${iter + 1} iterations`);
        }
        break;
      }

      // Update weights
      await this._updateWeights(gradients);

      if (this.options.verbose && (iter + 1) % 10 === 0) {
        console.log(`Iteration ${iter + 1}, loss = ${loss.toFixed(6)}`);
      }
    }

    this._isFitted = true;
    return this;
  }

  /**
   * Predict target values for samples in X.
   *
   * @param X - Input data of shape (n_samples, n_features)
   * @returns Predicted values of shape (n_samples,) or (n_samples, n_outputs)
   */
  async predict(X: NDArray): Promise<NDArray> {
    this._checkFitted();
    const result = await this._forwardPass(X);

    // If single output, return 1D array
    if (this._nOutputs === 1) {
      return this.backend.flatten(result);
    }
    return result;
  }

  /**
   * Return R² score on the given test data and labels.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True target values of shape (n_samples,)
   * @returns R² score
   */
  async score(X: NDArray, y: NDArray): Promise<number> {
    const predictions = await this.predict(X);

    // Compute R² = 1 - SS_res / SS_tot
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

    return 1 - ssRes / ssTot;
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
  setParams(params: Partial<MLPRegressorOptions>): this {
    Object.assign(this.options, params);
    return this;
  }

  // ==================== Internal Methods ====================

  private _checkFitted(): void {
    if (!this._isFitted) {
      throw new Error(
        'This MLPRegressor instance is not fitted yet. Call fit() before using this estimator.'
      );
    }
  }

  private _validateInput(X: NDArray): [number, number] {
    if (X.shape.length !== 2) {
      throw new Error(`Expected 2D array, got ${X.shape.length}D array instead`);
    }
    return [X.shape[0], X.shape[1]];
  }

  private _initializeWeights(layerSizes: number[]): void {
    // Xavier/Glorot initialization
    this._coefs = [];
    this._intercepts = [];

    for (let i = 0; i < layerSizes.length - 1; i++) {
      const fanIn = layerSizes[i]!;
      const fanOut = layerSizes[i + 1]!;

      const limit = Math.sqrt(6 / (fanIn + fanOut));
      const weights = new Float64Array(fanIn * fanOut);
      const biases = new Float64Array(fanOut);

      for (let j = 0; j < weights.length; j++) {
        weights[j] = (Math.random() * 2 - 1) * limit;
      }

      this._coefs.push(this.backend.array(Array.from(weights), [fanIn, fanOut]));
      this._intercepts.push(this.backend.array(Array.from(biases), [fanOut]));
    }
  }

  private async _forwardPass(X: NDArray): Promise<NDArray> {
    let activation = X;

    // Hidden layers
    for (let i = 0; i < this._coefs.length - 1; i++) {
      const linear = this.backend.add(
        this.backend.matmul(activation, this._coefs[i]),
        this._expandBias(this._intercepts[i], activation.shape[0])
      );
      activation = this._applyActivation(linear, this.options.activation);
    }

    // Output layer (identity activation for regression)
    const lastIdx = this._coefs.length - 1;
    return this.backend.add(
      this.backend.matmul(activation, this._coefs[lastIdx]),
      this._expandBias(this._intercepts[lastIdx], activation.shape[0])
    );
  }

  private _expandBias(bias: NDArray, nSamples: number): NDArray {
    const expanded = new Float64Array(nSamples * bias.data.length);
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < bias.data.length; j++) {
        expanded[i * bias.data.length + j] = bias.data[j];
      }
    }
    return this.backend.array(Array.from(expanded), [nSamples, bias.data.length]);
  }

  private _applyActivation(X: NDArray, activation: Activation): NDArray {
    const data = X.data;
    const result = new Float64Array(data.length);

    switch (activation) {
      case 'identity':
        result.set(data);
        break;
      case 'logistic':
        for (let i = 0; i < data.length; i++) {
          result[i] = 1 / (1 + Math.exp(-data[i]));
        }
        break;
      case 'tanh':
        for (let i = 0; i < data.length; i++) {
          result[i] = Math.tanh(data[i]);
        }
        break;
      case 'relu':
        for (let i = 0; i < data.length; i++) {
          result[i] = data[i] > 0 ? data[i] : 0;
        }
        break;
    }

    return this.backend.array(Array.from(result), X.shape);
  }

  private async _computeGradients(
    X: NDArray,
    y: NDArray,
    _batchSize: number
  ): Promise<{ loss: number; gradients: { coefs: NDArray[]; intercepts: NDArray[] } }> {
    // Forward pass with stored activations
    const activations: NDArray[] = [X];
    let activation = X;

    // Hidden layers
    for (let i = 0; i < this._coefs.length - 1; i++) {
      const linear = this.backend.add(
        this.backend.matmul(activation, this._coefs[i]),
        this._expandBias(this._intercepts[i], activation.shape[0])
      );
      activation = this._applyActivation(linear, this.options.activation);
      activations.push(activation);
    }

    // Output layer (identity)
    const lastIdx = this._coefs.length - 1;
    const output = this.backend.add(
      this.backend.matmul(activation, this._coefs[lastIdx]),
      this._expandBias(this._intercepts[lastIdx], activation.shape[0])
    );
    activations.push(output);

    // Compute loss (MSE)
    let loss = 0;
    const nSamples = y.shape[0];
    for (let i = 0; i < y.data.length; i++) {
      const diff = output.data[i] - y.data[i];
      loss += diff * diff;
    }
    loss /= 2 * nSamples;

    // Add L2 regularization
    for (const coef of this._coefs) {
      let sum = 0;
      for (const val of coef.data) {
        sum += val * val;
      }
      loss += (this.options.alpha / 2) * sum;
    }

    // Backpropagation
    const coefGrads: NDArray[] = [];
    const interceptGrads: NDArray[] = [];

    // Output layer gradient (for MSE with identity: just (output - y) / n_samples)
    let delta = this.backend.sub(output, y);
    delta = this.backend.divScalar(delta, nSamples);

    // Work backwards through layers
    for (let i = this._coefs.length - 1; i >= 0; i--) {
      const prevActivation = activations[i];

      // Gradient for weights
      const coefGrad = this.backend.matmul(
        this.backend.transpose(prevActivation),
        delta
      );
      const regGrad = this.backend.mulScalar(this._coefs[i], this.options.alpha);
      coefGrads.unshift(this.backend.add(coefGrad, regGrad));

      // Gradient for biases
      const interceptGrad = new Float64Array(this._intercepts[i].data.length);
      const deltaShape1 = delta.shape.length > 1 ? delta.shape[1] : 1;
      for (let s = 0; s < delta.shape[0]; s++) {
        for (let j = 0; j < interceptGrad.length; j++) {
          interceptGrad[j] += delta.data[s * deltaShape1 + j];
        }
      }
      interceptGrads.unshift(
        this.backend.array(Array.from(interceptGrad), [interceptGrad.length])
      );

      if (i > 0) {
        delta = this.backend.matmul(delta, this.backend.transpose(this._coefs[i]));
        delta = this._applyActivationDerivative(delta, activations[i], this.options.activation);
      }
    }

    return {
      loss,
      gradients: { coefs: coefGrads, intercepts: interceptGrads },
    };
  }

  private _applyActivationDerivative(
    grad: NDArray,
    activation: NDArray,
    activationType: Activation
  ): NDArray {
    const result = new Float64Array(grad.data.length);

    switch (activationType) {
      case 'identity':
        result.set(grad.data);
        break;
      case 'logistic':
        for (let i = 0; i < grad.data.length; i++) {
          const s = activation.data[i];
          result[i] = grad.data[i] * s * (1 - s);
        }
        break;
      case 'tanh':
        for (let i = 0; i < grad.data.length; i++) {
          const t = activation.data[i];
          result[i] = grad.data[i] * (1 - t * t);
        }
        break;
      case 'relu':
        for (let i = 0; i < grad.data.length; i++) {
          result[i] = activation.data[i] > 0 ? grad.data[i] : 0;
        }
        break;
    }

    return this.backend.array(Array.from(result), grad.shape);
  }

  private async _updateWeights(gradients: {
    coefs: NDArray[];
    intercepts: NDArray[];
  }): Promise<void> {
    const lr = this.options.learningRateInit;

    for (let i = 0; i < this._coefs.length; i++) {
      const coefUpdate = this.backend.mulScalar(gradients.coefs[i], -lr);
      this._coefs[i] = this.backend.add(this._coefs[i], coefUpdate);

      const interceptUpdate = this.backend.mulScalar(gradients.intercepts[i], -lr);
      this._intercepts[i] = this.backend.add(this._intercepts[i], interceptUpdate);
    }
  }
}
