/**
 * Regression metrics
 *
 * Metrics for evaluating regression models.
 * Matches sklearn.metrics regression functions 1:1.
 */

import type { NDArray } from 'numpyjs';

/**
 * Mean squared error regression loss.
 *
 * @param yTrue - Ground truth (correct) target values
 * @param yPred - Estimated target values
 * @param sampleWeight - Sample weights (optional)
 * @param squared - If True return MSE, if False return RMSE
 * @returns Mean squared error (or root mean squared error)
 *
 * @example
 * ```typescript
 * const yTrue = backend.array([3, -0.5, 2, 7]);
 * const yPred = backend.array([2.5, 0.0, 2, 8]);
 * const mse = meanSquaredError(yTrue, yPred);  // 0.375
 * ```
 */
export function meanSquaredError(
  yTrue: NDArray,
  yPred: NDArray,
  sampleWeight?: NDArray,
  squared: boolean = true
): number {
  if (yTrue.data.length !== yPred.data.length) {
    throw new Error(
      `Found input arrays with inconsistent numbers of samples: ` +
        `[${yTrue.data.length}, ${yPred.data.length}]`
    );
  }

  const n = yTrue.data.length;
  let sumSquaredError = 0;
  let totalWeight = 0;

  for (let i = 0; i < n; i++) {
    const weight = sampleWeight ? sampleWeight.data[i] : 1;
    const error = yTrue.data[i] - yPred.data[i];
    sumSquaredError += weight * error * error;
    totalWeight += weight;
  }

  const mse = totalWeight > 0 ? sumSquaredError / totalWeight : 0;
  return squared ? mse : Math.sqrt(mse);
}

/**
 * Root mean squared error regression loss.
 *
 * Equivalent to meanSquaredError with squared=false.
 *
 * @param yTrue - Ground truth (correct) target values
 * @param yPred - Estimated target values
 * @param sampleWeight - Sample weights (optional)
 * @returns Root mean squared error
 */
export function rootMeanSquaredError(
  yTrue: NDArray,
  yPred: NDArray,
  sampleWeight?: NDArray
): number {
  return meanSquaredError(yTrue, yPred, sampleWeight, false);
}

/**
 * Mean absolute error regression loss.
 *
 * @param yTrue - Ground truth (correct) target values
 * @param yPred - Estimated target values
 * @param sampleWeight - Sample weights (optional)
 * @returns Mean absolute error
 *
 * @example
 * ```typescript
 * const yTrue = backend.array([3, -0.5, 2, 7]);
 * const yPred = backend.array([2.5, 0.0, 2, 8]);
 * const mae = meanAbsoluteError(yTrue, yPred);  // 0.5
 * ```
 */
export function meanAbsoluteError(
  yTrue: NDArray,
  yPred: NDArray,
  sampleWeight?: NDArray
): number {
  if (yTrue.data.length !== yPred.data.length) {
    throw new Error(
      `Found input arrays with inconsistent numbers of samples: ` +
        `[${yTrue.data.length}, ${yPred.data.length}]`
    );
  }

  const n = yTrue.data.length;
  let sumAbsError = 0;
  let totalWeight = 0;

  for (let i = 0; i < n; i++) {
    const weight = sampleWeight ? sampleWeight.data[i] : 1;
    sumAbsError += weight * Math.abs(yTrue.data[i] - yPred.data[i]);
    totalWeight += weight;
  }

  return totalWeight > 0 ? sumAbsError / totalWeight : 0;
}

/**
 * R² (coefficient of determination) regression score function.
 *
 * Best possible score is 1.0 and it can be negative (because the model
 * can be arbitrarily worse). A constant model that always predicts the
 * expected value of y, disregarding the input features, would get R² = 0.
 *
 * R² = 1 - SS_res / SS_tot
 *
 * @param yTrue - Ground truth (correct) target values
 * @param yPred - Estimated target values
 * @param sampleWeight - Sample weights (optional)
 * @returns R² score
 *
 * @example
 * ```typescript
 * const yTrue = backend.array([3, -0.5, 2, 7]);
 * const yPred = backend.array([2.5, 0.0, 2, 8]);
 * const r2 = r2Score(yTrue, yPred);  // 0.948...
 * ```
 */
export function r2Score(
  yTrue: NDArray,
  yPred: NDArray,
  sampleWeight?: NDArray
): number {
  if (yTrue.data.length !== yPred.data.length) {
    throw new Error(
      `Found input arrays with inconsistent numbers of samples: ` +
        `[${yTrue.data.length}, ${yPred.data.length}]`
    );
  }

  const n = yTrue.data.length;

  // Compute weighted mean of y_true
  let yMean = 0;
  let totalWeight = 0;
  for (let i = 0; i < n; i++) {
    const weight = sampleWeight ? sampleWeight.data[i] : 1;
    yMean += weight * yTrue.data[i];
    totalWeight += weight;
  }
  yMean /= totalWeight;

  // Compute SS_res and SS_tot
  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < n; i++) {
    const weight = sampleWeight ? sampleWeight.data[i] : 1;
    const residual = yTrue.data[i] - yPred.data[i];
    const deviation = yTrue.data[i] - yMean;
    ssRes += weight * residual * residual;
    ssTot += weight * deviation * deviation;
  }

  // Handle edge case where SS_tot is 0
  if (ssTot === 0) {
    return ssRes === 0 ? 1.0 : 0.0;
  }

  return 1 - ssRes / ssTot;
}

/**
 * Mean absolute percentage error (MAPE) regression loss.
 *
 * Note: When y_true has zeros, MAPE can be undefined or very large.
 *
 * @param yTrue - Ground truth (correct) target values
 * @param yPred - Estimated target values
 * @param sampleWeight - Sample weights (optional)
 * @returns Mean absolute percentage error (as a fraction, not percentage)
 */
export function meanAbsolutePercentageError(
  yTrue: NDArray,
  yPred: NDArray,
  sampleWeight?: NDArray
): number {
  if (yTrue.data.length !== yPred.data.length) {
    throw new Error(
      `Found input arrays with inconsistent numbers of samples: ` +
        `[${yTrue.data.length}, ${yPred.data.length}]`
    );
  }

  const n = yTrue.data.length;
  let sumPercentError = 0;
  let totalWeight = 0;
  const eps = 1e-15; // Small value to avoid division by zero

  for (let i = 0; i < n; i++) {
    const weight = sampleWeight ? sampleWeight.data[i] : 1;
    const absTrue = Math.abs(yTrue.data[i]) + eps;
    sumPercentError +=
      weight * Math.abs((yTrue.data[i] - yPred.data[i]) / absTrue);
    totalWeight += weight;
  }

  return totalWeight > 0 ? sumPercentError / totalWeight : 0;
}

/**
 * Explained variance regression score function.
 *
 * Best possible score is 1.0, lower values are worse.
 *
 * @param yTrue - Ground truth (correct) target values
 * @param yPred - Estimated target values
 * @param sampleWeight - Sample weights (optional)
 * @returns Explained variance score
 */
export function explainedVarianceScore(
  yTrue: NDArray,
  yPred: NDArray,
  sampleWeight?: NDArray
): number {
  if (yTrue.data.length !== yPred.data.length) {
    throw new Error(
      `Found input arrays with inconsistent numbers of samples: ` +
        `[${yTrue.data.length}, ${yPred.data.length}]`
    );
  }

  const n = yTrue.data.length;

  // Compute residuals
  const residuals: number[] = [];
  for (let i = 0; i < n; i++) {
    residuals.push(yTrue.data[i] - yPred.data[i]);
  }

  // Compute variance of y_true
  let yMean = 0;
  let totalWeight = 0;
  for (let i = 0; i < n; i++) {
    const weight = sampleWeight ? sampleWeight.data[i] : 1;
    yMean += weight * yTrue.data[i];
    totalWeight += weight;
  }
  yMean /= totalWeight;

  let varY = 0;
  for (let i = 0; i < n; i++) {
    const weight = sampleWeight ? sampleWeight.data[i] : 1;
    const deviation = yTrue.data[i] - yMean;
    varY += weight * deviation * deviation;
  }
  varY /= totalWeight;

  // Compute variance of residuals
  let resMean = 0;
  for (let i = 0; i < n; i++) {
    const weight = sampleWeight ? sampleWeight.data[i] : 1;
    resMean += weight * residuals[i];
  }
  resMean /= totalWeight;

  let varRes = 0;
  for (let i = 0; i < n; i++) {
    const weight = sampleWeight ? sampleWeight.data[i] : 1;
    const deviation = residuals[i] - resMean;
    varRes += weight * deviation * deviation;
  }
  varRes /= totalWeight;

  // Handle edge case where variance of y is 0
  if (varY === 0) {
    return varRes === 0 ? 1.0 : 0.0;
  }

  return 1 - varRes / varY;
}
