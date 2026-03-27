/**
 * sklearn.linear_model module
 *
 * Linear models for classification and regression.
 * Matches sklearn.linear_model API 1:1.
 */

export { LinearRegression } from './linear_regression.js';
export { LogisticRegression } from './logistic_regression.js';

export type {
  LinearRegressionOptions,
  LogisticRegressionOptions,
  LinearRegressionAttributes,
  LogisticRegressionAttributes,
} from './types.js';
