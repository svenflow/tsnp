/**
 * Classification metrics
 *
 * Metrics for evaluating classification models.
 * Matches sklearn.metrics classification functions 1:1.
 */

import type { NDArray, Backend } from 'numpyjs';

/**
 * Accuracy classification score.
 *
 * In multilabel classification, this function computes subset accuracy:
 * the set of labels predicted for a sample must exactly match the
 * corresponding set of labels in y_true.
 *
 * @param yTrue - Ground truth (correct) labels
 * @param yPred - Predicted labels
 * @param normalize - If True, return fraction of correctly classified samples.
 *                    Otherwise, return the number of correctly classified samples.
 * @param sampleWeight - Sample weights (optional)
 * @returns Accuracy score
 *
 * @example
 * ```typescript
 * const yTrue = backend.array([0, 1, 2, 3]);
 * const yPred = backend.array([0, 2, 1, 3]);
 * const acc = accuracyScore(yTrue, yPred);  // 0.5
 * ```
 */
export function accuracyScore(
  yTrue: NDArray,
  yPred: NDArray,
  normalize = true,
  sampleWeight?: NDArray
): number {
  if (yTrue.data.length !== yPred.data.length) {
    throw new Error(
      `Found input arrays with inconsistent numbers of samples: ` +
        `[${yTrue.data.length}, ${yPred.data.length}]`
    );
  }

  const n = yTrue.data.length;
  let correct = 0;
  let totalWeight = 0;

  for (let i = 0; i < n; i++) {
    const weight = sampleWeight ? sampleWeight.data[i] : 1;
    if (yTrue.data[i] === yPred.data[i]) {
      correct += weight;
    }
    totalWeight += weight;
  }

  if (normalize) {
    return totalWeight > 0 ? correct / totalWeight : 0;
  }
  return correct;
}

/**
 * Compute precision score.
 *
 * The precision is the ratio tp / (tp + fp) where tp is the number of
 * true positives and fp the number of false positives.
 *
 * @param yTrue - Ground truth (correct) labels
 * @param yPred - Predicted labels
 * @param posLabel - The class to report (for binary classification)
 * @returns Precision score
 */
export function precisionScore(
  yTrue: NDArray,
  yPred: NDArray,
  posLabel: number = 1
): number {
  let tp = 0;
  let fp = 0;

  for (let i = 0; i < yTrue.data.length; i++) {
    if (yPred.data[i] === posLabel) {
      if (yTrue.data[i] === posLabel) {
        tp++;
      } else {
        fp++;
      }
    }
  }

  return tp + fp > 0 ? tp / (tp + fp) : 0;
}

/**
 * Compute recall score.
 *
 * The recall is the ratio tp / (tp + fn) where tp is the number of
 * true positives and fn the number of false negatives.
 *
 * @param yTrue - Ground truth (correct) labels
 * @param yPred - Predicted labels
 * @param posLabel - The class to report (for binary classification)
 * @returns Recall score
 */
export function recallScore(
  yTrue: NDArray,
  yPred: NDArray,
  posLabel: number = 1
): number {
  let tp = 0;
  let fn = 0;

  for (let i = 0; i < yTrue.data.length; i++) {
    if (yTrue.data[i] === posLabel) {
      if (yPred.data[i] === posLabel) {
        tp++;
      } else {
        fn++;
      }
    }
  }

  return tp + fn > 0 ? tp / (tp + fn) : 0;
}

/**
 * Compute F1 score.
 *
 * The F1 score is the harmonic mean of precision and recall.
 * F1 = 2 * (precision * recall) / (precision + recall)
 *
 * @param yTrue - Ground truth (correct) labels
 * @param yPred - Predicted labels
 * @param posLabel - The class to report (for binary classification)
 * @returns F1 score
 */
export function f1Score(
  yTrue: NDArray,
  yPred: NDArray,
  posLabel: number = 1
): number {
  const precision = precisionScore(yTrue, yPred, posLabel);
  const recall = recallScore(yTrue, yPred, posLabel);

  if (precision + recall === 0) {
    return 0;
  }

  return (2 * precision * recall) / (precision + recall);
}

/**
 * Compute confusion matrix.
 *
 * @param yTrue - Ground truth (correct) labels
 * @param yPred - Predicted labels
 * @param backend - Backend to use for array operations
 * @param labels - List of labels to index the matrix (optional)
 * @returns Confusion matrix as NDArray of shape (n_classes, n_classes)
 */
export function confusionMatrix(
  yTrue: NDArray,
  yPred: NDArray,
  backend: Backend,
  labels?: number[]
): NDArray {
  // Get unique labels if not provided
  const labelSet = new Set<number>();
  for (const val of yTrue.data) {
    labelSet.add(val);
  }
  for (const val of yPred.data) {
    labelSet.add(val);
  }
  const sortedLabels = labels || Array.from(labelSet).sort((a, b) => a - b);
  const nLabels = sortedLabels.length;

  // Create label to index mapping
  const labelToIdx = new Map<number, number>();
  sortedLabels.forEach((label, idx) => {
    labelToIdx.set(label, idx);
  });

  // Compute confusion matrix
  const matrix = new Float64Array(nLabels * nLabels);

  for (let i = 0; i < yTrue.data.length; i++) {
    const trueLabel = yTrue.data[i];
    const predLabel = yPred.data[i];
    const trueIdx = labelToIdx.get(trueLabel);
    const predIdx = labelToIdx.get(predLabel);

    if (trueIdx !== undefined && predIdx !== undefined) {
      matrix[trueIdx * nLabels + predIdx]++;
    }
  }

  return backend.array(Array.from(matrix), [nLabels, nLabels]);
}

/**
 * Compute the log loss (cross-entropy loss).
 *
 * @param yTrue - Ground truth (correct) labels
 * @param yProb - Predicted probabilities
 * @param eps - Small value to avoid log(0)
 * @returns Log loss value
 */
export function logLoss(
  yTrue: NDArray,
  yProb: NDArray,
  eps: number = 1e-15
): number {
  const n = yTrue.data.length;
  let loss = 0;

  for (let i = 0; i < n; i++) {
    const p = Math.max(eps, Math.min(1 - eps, yProb.data[i]));
    if (yTrue.data[i] === 1) {
      loss -= Math.log(p);
    } else {
      loss -= Math.log(1 - p);
    }
  }

  return loss / n;
}
