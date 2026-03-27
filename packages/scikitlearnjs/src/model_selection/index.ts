/**
 * sklearn.model_selection module
 *
 * Tools for model selection and evaluation.
 * Matches sklearn.model_selection API 1:1.
 */

export { trainTestSplit } from './split.js';

export type {
  TrainTestSplitOptions,
  TrainTestSplitResult,
  CrossValidationOptions,
} from './types.js';
