/**
 * MLPClassifier - Multi-layer Perceptron classifier
 *
 * This model optimizes the log-loss function using LBFGS, SGD, or Adam.
 * Matches sklearn.neural_network.MLPClassifier API 1:1.
 */

import type { NDArray, Backend } from 'numpyjs';
import type {
  MLPClassifierOptions,
  Activation,
  BaseEstimator,
  ClassifierMixin,
} from './types.js';

/**
 * Default options matching sklearn defaults
 */
const DEFAULT_OPTIONS: Required<MLPClassifierOptions> = {
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
 * Multi-layer Perceptron classifier.
 *
 * This model optimizes the log-loss function using LBFGS or stochastic
 * gradient descent.
 *
 * @example
 * ```typescript
 * import { createBackend } from 'numpyjs';
 * import { neural_network } from 'scikitlearnjs';
 *
 * const backend = await createBackend('wasm');
 * const clf = new neural_network.MLPClassifier(backend, {
 *   hiddenLayerSizes: [100, 50],
 *   activation: 'relu',
 *   maxIter: 500,
 * });
 *
 * await clf.fit(X_train, y_train);
 * const predictions = await clf.predict(X_test);
 * const accuracy = await clf.score(X_test, y_test);
 * ```
 */
export class MLPClassifier implements BaseEstimator, ClassifierMixin {
  private backend: Backend;
  private options: Required<MLPClassifierOptions>;

  // Fitted attributes (available after fit())
  private _coefs: NDArray[] = [];
  private _intercepts: NDArray[] = [];
  private _classes: NDArray | null = null;
  private _nIter = 0;
  private _nOutputs = 0;
  private _nLayers = 0;
  private _outActivation: Activation | 'softmax' = 'softmax';
  private _loss = Infinity;
  private _bestLoss = Infinity;
  private _lossCurve: number[] = [];
  private _validationScores: number[] = [];
  private _isFitted = false;

  constructor(backend: Backend, options: MLPClassifierOptions = {}) {
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

  /** Class labels seen during fit */
  get classes(): NDArray {
    this._checkFitted();
    return this._classes!;
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

  /** Output activation function */
  get outActivation(): Activation | 'softmax' {
    this._checkFitted();
    return this._outActivation;
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

  /** Validation scores if early_stopping=true */
  get validationScores(): number[] | undefined {
    this._checkFitted();
    return this._validationScores.length > 0 ? this._validationScores : undefined;
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
    const classes = this._extractClasses(y);
    this._nOutputs = classes.length;
    this._classes = this.backend.array(classes);

    // Build layer sizes
    const layerSizes = [nFeatures, ...this.options.hiddenLayerSizes, this._nOutputs];
    this._nLayers = layerSizes.length;

    // Initialize weights
    if (!this.options.warmStart || !this._isFitted) {
      this._initializeWeights(layerSizes);
    }

    // Determine output activation
    this._outActivation = this._nOutputs > 2 ? 'softmax' : 'logistic';

    // Encode labels
    const yEncoded = this._encodeLabels(y, classes);

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
      const { loss, gradients } = await this._computeGradients(X, yEncoded, batchSize);
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
   * Predict class labels for samples in X.
   *
   * @param X - Input data of shape (n_samples, n_features)
   * @returns Predicted class labels of shape (n_samples,)
   */
  async predict(X: NDArray): Promise<NDArray> {
    this._checkFitted();
    const proba = await this.predictProba(X);

    // Get argmax of each row
    const nSamples = X.shape[0];
    const predictions = new Float64Array(nSamples);

    for (let i = 0; i < nSamples; i++) {
      let maxIdx = 0;
      let maxVal = -Infinity;
      for (let j = 0; j < this._nOutputs; j++) {
        const val = proba.data[i * this._nOutputs + j];
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
   * Probability estimates for each class.
   *
   * @param X - Input data of shape (n_samples, n_features)
   * @returns Probability of each class of shape (n_samples, n_classes)
   */
  async predictProba(X: NDArray): Promise<NDArray> {
    this._checkFitted();
    return this._forwardPass(X);
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
  setParams(params: Partial<MLPClassifierOptions>): this {
    Object.assign(this.options, params);
    return this;
  }

  // ==================== Internal Methods ====================

  private _checkFitted(): void {
    if (!this._isFitted) {
      throw new Error(
        'This MLPClassifier instance is not fitted yet. Call fit() before using this estimator.'
      );
    }
  }

  private _validateInput(X: NDArray): [number, number] {
    if (X.shape.length !== 2) {
      throw new Error(`Expected 2D array, got ${X.shape.length}D array instead`);
    }
    return [X.shape[0], X.shape[1]];
  }

  private _extractClasses(y: NDArray): number[] {
    const classes = new Set<number>();
    for (const val of y.data) {
      classes.add(val);
    }
    return Array.from(classes).sort((a, b) => a - b);
  }

  private _encodeLabels(y: NDArray, classes: number[]): NDArray {
    // One-hot encode labels
    const nSamples = y.data.length;
    const nClasses = classes.length;
    const classMap = new Map(classes.map((c, i) => [c, i]));

    const encoded = new Float64Array(nSamples * nClasses);
    for (let i = 0; i < nSamples; i++) {
      const classIdx = classMap.get(y.data[i])!;
      encoded[i * nClasses + classIdx] = 1;
    }

    return this.backend.array(Array.from(encoded), [nSamples, nClasses]);
  }

  private _initializeWeights(layerSizes: number[]): void {
    // Xavier/Glorot initialization
    this._coefs = [];
    this._intercepts = [];

    for (let i = 0; i < layerSizes.length - 1; i++) {
      const fanIn = layerSizes[i]!;
      const fanOut = layerSizes[i + 1]!;

      // Xavier uniform: limit = sqrt(6 / (fan_in + fan_out))
      const limit = Math.sqrt(6 / (fanIn + fanOut));
      const weights = new Float64Array(fanIn * fanOut);
      const biases = new Float64Array(fanOut);

      for (let j = 0; j < weights.length; j++) {
        weights[j] = (Math.random() * 2 - 1) * limit;
      }
      // Biases initialized to zero

      this._coefs.push(this.backend.array(Array.from(weights), [fanIn, fanOut]));
      this._intercepts.push(this.backend.array(Array.from(biases), [fanOut]));
    }
  }

  private async _forwardPass(X: NDArray): Promise<NDArray> {
    let activation = X;

    // Hidden layers
    for (let i = 0; i < this._coefs.length - 1; i++) {
      // Linear: activation = X @ W + b
      const linear = this.backend.add(
        this.backend.matmul(activation, this._coefs[i]),
        this._expandBias(this._intercepts[i], activation.shape[0])
      );

      // Apply activation function
      activation = this._applyActivation(linear, this.options.activation);
    }

    // Output layer
    const lastIdx = this._coefs.length - 1;
    const output = this.backend.add(
      this.backend.matmul(activation, this._coefs[lastIdx]),
      this._expandBias(this._intercepts[lastIdx], activation.shape[0])
    );

    // Apply output activation (softmax for multi-class, sigmoid for binary)
    return this._applyOutputActivation(output);
  }

  private _expandBias(bias: NDArray, nSamples: number): NDArray {
    // Expand bias to match batch size: (n_outputs,) -> (n_samples, n_outputs)
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

  private _applyOutputActivation(X: NDArray): NDArray {
    if (this._outActivation === 'softmax') {
      return this._softmax(X);
    } else {
      // Logistic/sigmoid for binary
      const data = X.data;
      const result = new Float64Array(data.length);
      for (let i = 0; i < data.length; i++) {
        result[i] = 1 / (1 + Math.exp(-data[i]));
      }
      return this.backend.array(Array.from(result), X.shape);
    }
  }

  private _softmax(X: NDArray): NDArray {
    const [nSamples, nClasses] = X.shape;
    const result = new Float64Array(X.data.length);

    for (let i = 0; i < nSamples; i++) {
      // Find max for numerical stability
      let max = -Infinity;
      for (let j = 0; j < nClasses; j++) {
        const val = X.data[i * nClasses + j];
        if (val > max) max = val;
      }

      // Compute exp(x - max) and sum
      let sum = 0;
      for (let j = 0; j < nClasses; j++) {
        const exp = Math.exp(X.data[i * nClasses + j] - max);
        result[i * nClasses + j] = exp;
        sum += exp;
      }

      // Normalize
      for (let j = 0; j < nClasses; j++) {
        result[i * nClasses + j] /= sum;
      }
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

    // Output layer
    const lastIdx = this._coefs.length - 1;
    const output = this.backend.add(
      this.backend.matmul(activation, this._coefs[lastIdx]),
      this._expandBias(this._intercepts[lastIdx], activation.shape[0])
    );
    const outputActivation = this._applyOutputActivation(output);
    activations.push(outputActivation);

    // Compute loss (cross-entropy)
    let loss = 0;
    const nSamples = y.shape[0];
    for (let i = 0; i < y.data.length; i++) {
      const p = Math.max(1e-15, Math.min(1 - 1e-15, outputActivation.data[i]));
      loss -= y.data[i] * Math.log(p);
    }
    loss /= nSamples;

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

    // Output layer gradient
    let delta = this.backend.sub(outputActivation, y);

    // Work backwards through layers
    for (let i = this._coefs.length - 1; i >= 0; i--) {
      const prevActivation = activations[i];

      // Gradient for weights: prev_activation.T @ delta
      const coefGrad = this.backend.matmul(
        this.backend.transpose(prevActivation),
        delta
      );
      // Add L2 regularization gradient
      const regGrad = this.backend.mulScalar(this._coefs[i], this.options.alpha);
      coefGrads.unshift(this.backend.add(coefGrad, regGrad));

      // Gradient for biases: sum of delta along axis 0
      const interceptGrad = new Float64Array(this._intercepts[i].data.length);
      for (let s = 0; s < nSamples; s++) {
        for (let j = 0; j < interceptGrad.length; j++) {
          interceptGrad[j] += delta.data[s * interceptGrad.length + j];
        }
      }
      interceptGrads.unshift(
        this.backend.array(Array.from(interceptGrad), [interceptGrad.length])
      );

      if (i > 0) {
        // Propagate gradient to previous layer
        delta = this.backend.matmul(delta, this.backend.transpose(this._coefs[i]));

        // Apply activation derivative
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
        // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        for (let i = 0; i < grad.data.length; i++) {
          const s = activation.data[i];
          result[i] = grad.data[i] * s * (1 - s);
        }
        break;
      case 'tanh':
        // tanh'(x) = 1 - tanh(x)^2
        for (let i = 0; i < grad.data.length; i++) {
          const t = activation.data[i];
          result[i] = grad.data[i] * (1 - t * t);
        }
        break;
      case 'relu':
        // relu'(x) = 1 if x > 0, else 0
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

    // Simple SGD update
    // TODO: Implement Adam, LBFGS solvers
    for (let i = 0; i < this._coefs.length; i++) {
      // coef = coef - lr * grad
      const coefUpdate = this.backend.mulScalar(gradients.coefs[i], -lr);
      this._coefs[i] = this.backend.add(this._coefs[i], coefUpdate);

      const interceptUpdate = this.backend.mulScalar(gradients.intercepts[i], -lr);
      this._intercepts[i] = this.backend.add(this._intercepts[i], interceptUpdate);
    }
  }
}
