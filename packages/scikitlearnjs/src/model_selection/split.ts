/**
 * Dataset splitting utilities
 *
 * Functions for splitting data into train/test sets.
 * Matches sklearn.model_selection API 1:1.
 */

import type { NDArray, Backend } from 'numpyjs';
import type { TrainTestSplitOptions } from './types.js';

/**
 * Simple PRNG (Mulberry32) for reproducible shuffling
 */
function mulberry32(seed: number): () => number {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Fisher-Yates shuffle with optional seed
 */
function shuffleArray(arr: number[], seed?: number): number[] {
  const result = [...arr];
  const random = seed !== undefined ? mulberry32(seed) : Math.random;

  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }

  return result;
}

/**
 * Split arrays or matrices into random train and test subsets.
 *
 * Quick utility that wraps input validation, shuffle, and splitting.
 *
 * @param X - Features array of shape (n_samples, n_features)
 * @param y - Target array of shape (n_samples,) - optional
 * @param backend - Backend to use for array operations
 * @param options - Split options
 * @returns Object containing XTrain, XTest, and optionally yTrain, yTest
 *
 * @example
 * ```typescript
 * const { XTrain, XTest, yTrain, yTest } = trainTestSplit(X, y, backend, {
 *   testSize: 0.2,
 *   randomState: 42,
 * });
 * ```
 */
export function trainTestSplit(
  X: NDArray,
  y: NDArray | undefined,
  backend: Backend,
  options: TrainTestSplitOptions = {}
): { XTrain: NDArray; XTest: NDArray; yTrain?: NDArray; yTest?: NDArray } {
  const {
    testSize = 0.25,
    trainSize,
    randomState,
    shuffle = true,
    stratify,
  } = options;

  // Validate inputs
  if (X.shape.length !== 2) {
    throw new Error(`Expected 2D array for X, got ${X.shape.length}D`);
  }

  const nSamples = X.shape[0];
  const nFeatures = X.shape[1];

  if (y && y.data.length !== nSamples) {
    throw new Error(
      `X and y have inconsistent numbers of samples: ${nSamples} vs ${y.data.length}`
    );
  }

  // Determine split sizes
  let nTest: number;
  let nTrain: number;

  if (testSize !== undefined && trainSize !== undefined) {
    nTest = Math.floor(nSamples * testSize);
    nTrain = Math.floor(nSamples * trainSize);
    if (nTest + nTrain > nSamples) {
      throw new Error('The sum of train_size and test_size exceeds 1.0');
    }
  } else if (testSize !== undefined) {
    nTest = Math.floor(nSamples * testSize);
    nTrain = nSamples - nTest;
  } else if (trainSize !== undefined) {
    nTrain = Math.floor(nSamples * trainSize);
    nTest = nSamples - nTrain;
  } else {
    nTest = Math.floor(nSamples * 0.25);
    nTrain = nSamples - nTest;
  }

  // Generate indices
  let indices = Array.from({ length: nSamples }, (_, i) => i);

  // Stratified split if requested
  if (stratify) {
    return stratifiedSplit(X, y, backend, stratify, nTrain, nTest, randomState);
  }

  // Shuffle if requested
  if (shuffle) {
    indices = shuffleArray(indices, randomState);
  }

  const trainIndices = indices.slice(0, nTrain);
  const testIndices = indices.slice(nTrain, nTrain + nTest);

  // Build output arrays
  const XTrainData = new Float64Array(nTrain * nFeatures);
  const XTestData = new Float64Array(nTest * nFeatures);

  for (let i = 0; i < nTrain; i++) {
    const srcIdx = trainIndices[i];
    for (let j = 0; j < nFeatures; j++) {
      XTrainData[i * nFeatures + j] = X.data[srcIdx * nFeatures + j];
    }
  }

  for (let i = 0; i < nTest; i++) {
    const srcIdx = testIndices[i];
    for (let j = 0; j < nFeatures; j++) {
      XTestData[i * nFeatures + j] = X.data[srcIdx * nFeatures + j];
    }
  }

  const result: {
    XTrain: NDArray;
    XTest: NDArray;
    yTrain?: NDArray;
    yTest?: NDArray;
  } = {
    XTrain: backend.array(Array.from(XTrainData), [nTrain, nFeatures]),
    XTest: backend.array(Array.from(XTestData), [nTest, nFeatures]),
  };

  // Split y if provided
  if (y) {
    const yTrainData = new Float64Array(nTrain);
    const yTestData = new Float64Array(nTest);

    for (let i = 0; i < nTrain; i++) {
      yTrainData[i] = y.data[trainIndices[i]];
    }

    for (let i = 0; i < nTest; i++) {
      yTestData[i] = y.data[testIndices[i]];
    }

    result.yTrain = backend.array(Array.from(yTrainData), [nTrain]);
    result.yTest = backend.array(Array.from(yTestData), [nTest]);
  }

  return result;
}

/**
 * Stratified train/test split
 */
function stratifiedSplit(
  X: NDArray,
  y: NDArray | undefined,
  backend: Backend,
  stratify: NDArray,
  nTrain: number,
  nTest: number,
  randomState?: number
): { XTrain: NDArray; XTest: NDArray; yTrain?: NDArray; yTest?: NDArray } {
  const nSamples = X.shape[0];
  const nFeatures = X.shape[1];

  // Group indices by class
  const classIndices = new Map<number, number[]>();
  for (let i = 0; i < nSamples; i++) {
    const cls = stratify.data[i];
    if (!classIndices.has(cls)) {
      classIndices.set(cls, []);
    }
    classIndices.get(cls)!.push(i);
  }

  // Shuffle each class's indices
  const trainIndices: number[] = [];
  const testIndices: number[] = [];

  const totalSamples = nTrain + nTest;
  const testRatio = nTest / totalSamples;

  for (const [, indices] of classIndices) {
    const shuffled = shuffleArray(indices, randomState);
    const classNTest = Math.round(shuffled.length * testRatio);
    const classNTrain = shuffled.length - classNTest;

    trainIndices.push(...shuffled.slice(0, classNTrain));
    testIndices.push(...shuffled.slice(classNTrain));
  }

  // Shuffle again
  const finalTrainIndices = shuffleArray(trainIndices, randomState);
  const finalTestIndices = shuffleArray(testIndices, randomState);

  // Build output arrays
  const actualNTrain = finalTrainIndices.length;
  const actualNTest = finalTestIndices.length;

  const XTrainData = new Float64Array(actualNTrain * nFeatures);
  const XTestData = new Float64Array(actualNTest * nFeatures);

  for (let i = 0; i < actualNTrain; i++) {
    const srcIdx = finalTrainIndices[i];
    for (let j = 0; j < nFeatures; j++) {
      XTrainData[i * nFeatures + j] = X.data[srcIdx * nFeatures + j];
    }
  }

  for (let i = 0; i < actualNTest; i++) {
    const srcIdx = finalTestIndices[i];
    for (let j = 0; j < nFeatures; j++) {
      XTestData[i * nFeatures + j] = X.data[srcIdx * nFeatures + j];
    }
  }

  const result: {
    XTrain: NDArray;
    XTest: NDArray;
    yTrain?: NDArray;
    yTest?: NDArray;
  } = {
    XTrain: backend.array(Array.from(XTrainData), [actualNTrain, nFeatures]),
    XTest: backend.array(Array.from(XTestData), [actualNTest, nFeatures]),
  };

  // Split y if provided
  if (y) {
    const yTrainData = new Float64Array(actualNTrain);
    const yTestData = new Float64Array(actualNTest);

    for (let i = 0; i < actualNTrain; i++) {
      yTrainData[i] = y.data[finalTrainIndices[i]];
    }

    for (let i = 0; i < actualNTest; i++) {
      yTestData[i] = y.data[finalTestIndices[i]];
    }

    result.yTrain = backend.array(Array.from(yTrainData), [actualNTrain]);
    result.yTest = backend.array(Array.from(yTestData), [actualNTest]);
  }

  return result;
}
