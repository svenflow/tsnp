/**
 * Types for sklearn.cluster module
 *
 * Matches sklearn.cluster API 1:1
 */

import type { NDArray } from 'numpyjs';

/**
 * Initialization methods for cluster centroids
 */
export type InitMethod = 'k-means++' | 'random' | NDArray;

/**
 * Options for KMeans
 */
export interface KMeansOptions {
  /** Number of clusters. Default: 8 */
  nClusters?: number;
  /** Method for initialization. Default: 'k-means++' */
  init?: InitMethod;
  /** Number of times to run with different centroid seeds. Default: 10 */
  nInit?: number;
  /** Maximum iterations. Default: 300 */
  maxIter?: number;
  /** Tolerance for convergence. Default: 1e-4 */
  tol?: number;
  /** Random state seed. Default: undefined */
  randomState?: number;
  /** Algorithm to use. Default: 'lloyd' */
  algorithm?: 'lloyd' | 'elkan';
  /** Verbosity mode. Default: 0 */
  verbose?: number;
}

/**
 * Fitted attributes for KMeans
 */
export interface KMeansAttributes {
  /** Coordinates of cluster centers */
  clusterCenters: NDArray;
  /** Labels of each point */
  labels: NDArray;
  /** Sum of squared distances to closest cluster center */
  inertia: number;
  /** Number of iterations run */
  nIter: number;
  /** Number of features seen during fit */
  nFeaturesIn: number;
}
