/**
 * sklearn.preprocessing module
 *
 * Data preprocessing utilities including scaling, normalization, and encoding.
 * Matches sklearn.preprocessing API 1:1.
 */

export { StandardScaler } from './standard_scaler.js';
export { MinMaxScaler } from './min_max_scaler.js';
export type {
  StandardScalerOptions,
  MinMaxScalerOptions,
  StandardScalerAttributes,
  MinMaxScalerAttributes,
  TransformerMixin,
} from './types.js';
