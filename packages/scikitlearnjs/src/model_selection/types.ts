/**
 * Types for sklearn.model_selection module
 *
 * Matches sklearn.model_selection API 1:1
 */

import type { NDArray } from 'numpyjs';

/**
 * Options for train_test_split
 */
export interface TrainTestSplitOptions {
  /** Proportion of dataset to include in test split (0.0 to 1.0) */
  testSize?: number;
  /** Proportion of dataset to include in train split (0.0 to 1.0) */
  trainSize?: number;
  /** Random seed for shuffling */
  randomState?: number;
  /** Whether to shuffle data before splitting */
  shuffle?: boolean;
  /** If not None, split in a stratified fashion using this as class labels */
  stratify?: NDArray;
}

/**
 * Result of train_test_split
 */
export interface TrainTestSplitResult {
  XTrain: NDArray;
  XTest: NDArray;
  yTrain?: NDArray;
  yTest?: NDArray;
}

/**
 * Options for cross-validation
 */
export interface CrossValidationOptions {
  /** Number of folds */
  cv?: number;
  /** Scoring function name */
  scoring?: string;
  /** Random seed */
  randomState?: number;
}
