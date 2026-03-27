/**
 * KMeans - K-Means clustering.
 *
 * K-Means clustering algorithm that aims to partition n observations into
 * k clusters in which each observation belongs to the cluster with the
 * nearest mean (cluster centers or cluster centroid).
 *
 * Matches sklearn.cluster.KMeans API 1:1.
 */

import type { NDArray, Backend } from 'numpyjs';
import type { KMeansOptions, InitMethod } from './types.js';

/**
 * Default options matching sklearn defaults
 */
const DEFAULT_OPTIONS: Required<KMeansOptions> = {
  nClusters: 8,
  init: 'k-means++',
  nInit: 10,
  maxIter: 300,
  tol: 1e-4,
  randomState: undefined as unknown as number,
  algorithm: 'lloyd',
  verbose: 0,
};

/**
 * Simple PRNG (Mulberry32) for reproducible initialization
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
 * K-Means clustering.
 *
 * @example
 * ```typescript
 * import { createBackend } from 'numpyjs';
 * import { cluster } from 'scikitlearnjs';
 *
 * const backend = await createBackend('wasm');
 * const kmeans = new cluster.KMeans(backend, { nClusters: 3 });
 *
 * await kmeans.fit(X);
 * const labels = kmeans.labels;
 * const centers = kmeans.clusterCenters;
 *
 * // Predict cluster for new samples
 * const newLabels = await kmeans.predict(X_new);
 * ```
 */
export class KMeans {
  private backend: Backend;
  private options: Required<KMeansOptions>;

  // Fitted attributes
  private _clusterCenters: NDArray | null = null;
  private _labels: NDArray | null = null;
  private _inertia = Infinity;
  private _nIter = 0;
  private _nFeaturesIn = 0;
  private _isFitted = false;

  constructor(backend: Backend, options: KMeansOptions = {}) {
    this.backend = backend;
    this.options = { ...DEFAULT_OPTIONS, ...options };
  }

  // ==================== Fitted Attributes ====================

  /** Coordinates of cluster centers, shape (n_clusters, n_features) */
  get clusterCenters(): NDArray {
    this._checkFitted();
    return this._clusterCenters!;
  }

  /** Labels of each point, shape (n_samples,) */
  get labels(): NDArray {
    this._checkFitted();
    return this._labels!;
  }

  /** Sum of squared distances of samples to their closest cluster center */
  get inertia(): number {
    this._checkFitted();
    return this._inertia;
  }

  /** Number of iterations run */
  get nIter(): number {
    this._checkFitted();
    return this._nIter;
  }

  /** Number of features seen during fit */
  get nFeaturesIn(): number {
    this._checkFitted();
    return this._nFeaturesIn;
  }

  // ==================== sklearn API Methods ====================

  /**
   * Compute k-means clustering.
   *
   * @param X - Training instances of shape (n_samples, n_features)
   * @returns The fitted estimator
   */
  async fit(X: NDArray): Promise<this> {
    this._validateInput(X);
    const [nSamples, nFeatures] = X.shape;
    this._nFeaturesIn = nFeatures;

    let bestInertia = Infinity;
    let bestCenters: NDArray | null = null;
    let bestLabels: NDArray | null = null;
    let bestNIter = 0;

    // Run multiple times with different initializations
    for (let init = 0; init < this.options.nInit; init++) {
      const seed =
        this.options.randomState !== undefined
          ? this.options.randomState + init
          : undefined;

      // Initialize centers
      let centers: NDArray;
      if (typeof this.options.init === 'object') {
        centers = this.options.init;
      } else if (this.options.init === 'k-means++') {
        centers = await this._initKMeansPlusPlus(X, seed);
      } else {
        centers = await this._initRandom(X, seed);
      }

      // Run Lloyd's algorithm
      const result = await this._lloydIteration(X, centers);

      if (result.inertia < bestInertia) {
        bestInertia = result.inertia;
        bestCenters = result.centers;
        bestLabels = result.labels;
        bestNIter = result.nIter;
      }
    }

    this._clusterCenters = bestCenters;
    this._labels = bestLabels;
    this._inertia = bestInertia;
    this._nIter = bestNIter;
    this._isFitted = true;

    return this;
  }

  /**
   * Compute cluster centers and predict cluster index for each sample.
   *
   * @param X - New data to transform, shape (n_samples, n_features)
   * @returns Predicted cluster index for each sample
   */
  async fitPredict(X: NDArray): Promise<NDArray> {
    await this.fit(X);
    return this._labels!;
  }

  /**
   * Predict the closest cluster each sample in X belongs to.
   *
   * @param X - New data to predict, shape (n_samples, n_features)
   * @returns Predicted cluster index for each sample
   */
  async predict(X: NDArray): Promise<NDArray> {
    this._checkFitted();
    this._validateInput(X);

    const nSamples = X.shape[0];
    const nFeatures = X.shape[1];
    const nClusters = this.options.nClusters;

    if (nFeatures !== this._nFeaturesIn) {
      throw new Error(
        `X has ${nFeatures} features, but KMeans was fitted with ${this._nFeaturesIn} features`
      );
    }

    const labels = new Float64Array(nSamples);

    for (let i = 0; i < nSamples; i++) {
      let minDist = Infinity;
      let minCluster = 0;

      for (let c = 0; c < nClusters; c++) {
        let dist = 0;
        for (let j = 0; j < nFeatures; j++) {
          const diff =
            X.data[i * nFeatures + j] -
            this._clusterCenters!.data[c * nFeatures + j];
          dist += diff * diff;
        }

        if (dist < minDist) {
          minDist = dist;
          minCluster = c;
        }
      }

      labels[i] = minCluster;
    }

    return this.backend.array(Array.from(labels), [nSamples]);
  }

  /**
   * Transform X to a cluster-distance space.
   *
   * @param X - New data to transform, shape (n_samples, n_features)
   * @returns Distances to cluster centers, shape (n_samples, n_clusters)
   */
  async transform(X: NDArray): Promise<NDArray> {
    this._checkFitted();
    this._validateInput(X);

    const nSamples = X.shape[0];
    const nFeatures = X.shape[1];
    const nClusters = this.options.nClusters;

    const distances = new Float64Array(nSamples * nClusters);

    for (let i = 0; i < nSamples; i++) {
      for (let c = 0; c < nClusters; c++) {
        let dist = 0;
        for (let j = 0; j < nFeatures; j++) {
          const diff =
            X.data[i * nFeatures + j] -
            this._clusterCenters!.data[c * nFeatures + j];
          dist += diff * diff;
        }
        distances[i * nClusters + c] = Math.sqrt(dist);
      }
    }

    return this.backend.array(Array.from(distances), [nSamples, nClusters]);
  }

  /**
   * Compute clustering and transform X to cluster-distance space.
   *
   * @param X - New data to fit and transform
   * @returns Distances to cluster centers
   */
  async fitTransform(X: NDArray): Promise<NDArray> {
    await this.fit(X);
    return this.transform(X);
  }

  /**
   * Opposite of the value of X on the K-means objective (inertia).
   *
   * @param X - New data to score
   * @returns Negative inertia
   */
  async score(X: NDArray): Promise<number> {
    this._checkFitted();
    const labels = await this.predict(X);

    const nSamples = X.shape[0];
    const nFeatures = X.shape[1];
    let inertia = 0;

    for (let i = 0; i < nSamples; i++) {
      const c = labels.data[i];
      for (let j = 0; j < nFeatures; j++) {
        const diff =
          X.data[i * nFeatures + j] -
          this._clusterCenters!.data[c * nFeatures + j];
        inertia += diff * diff;
      }
    }

    return -inertia;
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
  setParams(params: Partial<KMeansOptions>): this {
    Object.assign(this.options, params);
    return this;
  }

  // ==================== Internal Methods ====================

  private _checkFitted(): void {
    if (!this._isFitted) {
      throw new Error(
        'This KMeans instance is not fitted yet. Call fit() before using this estimator.'
      );
    }
  }

  private _validateInput(X: NDArray): void {
    if (X.shape.length !== 2) {
      throw new Error(`Expected 2D array, got ${X.shape.length}D array instead`);
    }
  }

  /**
   * K-means++ initialization
   */
  private async _initKMeansPlusPlus(
    X: NDArray,
    seed?: number
  ): Promise<NDArray> {
    const nSamples = X.shape[0];
    const nFeatures = X.shape[1];
    const nClusters = this.options.nClusters;

    const random = seed !== undefined ? mulberry32(seed) : Math.random;
    const centers = new Float64Array(nClusters * nFeatures);

    // Choose first center randomly
    const firstIdx = Math.floor(random() * nSamples);
    for (let j = 0; j < nFeatures; j++) {
      centers[j] = X.data[firstIdx * nFeatures + j];
    }

    // Choose remaining centers
    for (let c = 1; c < nClusters; c++) {
      // Compute squared distances to nearest center
      const distances = new Float64Array(nSamples);
      let totalDist = 0;

      for (let i = 0; i < nSamples; i++) {
        let minDist = Infinity;
        for (let prevC = 0; prevC < c; prevC++) {
          let dist = 0;
          for (let j = 0; j < nFeatures; j++) {
            const diff =
              X.data[i * nFeatures + j] - centers[prevC * nFeatures + j];
            dist += diff * diff;
          }
          if (dist < minDist) minDist = dist;
        }
        distances[i] = minDist;
        totalDist += minDist;
      }

      // Sample proportionally to squared distance
      let r = random() * totalDist;
      let nextIdx = 0;
      for (let i = 0; i < nSamples; i++) {
        r -= distances[i];
        if (r <= 0) {
          nextIdx = i;
          break;
        }
      }

      for (let j = 0; j < nFeatures; j++) {
        centers[c * nFeatures + j] = X.data[nextIdx * nFeatures + j];
      }
    }

    return this.backend.array(Array.from(centers), [nClusters, nFeatures]);
  }

  /**
   * Random initialization
   */
  private async _initRandom(X: NDArray, seed?: number): Promise<NDArray> {
    const nSamples = X.shape[0];
    const nFeatures = X.shape[1];
    const nClusters = this.options.nClusters;

    const random = seed !== undefined ? mulberry32(seed) : Math.random;

    // Sample without replacement
    const indices: number[] = [];
    const used = new Set<number>();

    while (indices.length < nClusters) {
      const idx = Math.floor(random() * nSamples);
      if (!used.has(idx)) {
        used.add(idx);
        indices.push(idx);
      }
    }

    const centers = new Float64Array(nClusters * nFeatures);
    for (let c = 0; c < nClusters; c++) {
      for (let j = 0; j < nFeatures; j++) {
        centers[c * nFeatures + j] = X.data[indices[c] * nFeatures + j];
      }
    }

    return this.backend.array(Array.from(centers), [nClusters, nFeatures]);
  }

  /**
   * Lloyd's algorithm iteration
   */
  private async _lloydIteration(
    X: NDArray,
    initialCenters: NDArray
  ): Promise<{
    centers: NDArray;
    labels: NDArray;
    inertia: number;
    nIter: number;
  }> {
    const nSamples = X.shape[0];
    const nFeatures = X.shape[1];
    const nClusters = this.options.nClusters;

    let centers = initialCenters;
    let labels = new Float64Array(nSamples);
    let inertia = Infinity;
    let nIter = 0;

    for (let iter = 0; iter < this.options.maxIter; iter++) {
      nIter = iter + 1;

      // Assignment step
      const newLabels = new Float64Array(nSamples);
      let newInertia = 0;

      for (let i = 0; i < nSamples; i++) {
        let minDist = Infinity;
        let minCluster = 0;

        for (let c = 0; c < nClusters; c++) {
          let dist = 0;
          for (let j = 0; j < nFeatures; j++) {
            const diff =
              X.data[i * nFeatures + j] - centers.data[c * nFeatures + j];
            dist += diff * diff;
          }

          if (dist < minDist) {
            minDist = dist;
            minCluster = c;
          }
        }

        newLabels[i] = minCluster;
        newInertia += minDist;
      }

      // Check convergence
      if (Math.abs(inertia - newInertia) < this.options.tol) {
        labels = newLabels;
        inertia = newInertia;
        break;
      }

      labels = newLabels;
      inertia = newInertia;

      // Update step
      const newCenters = new Float64Array(nClusters * nFeatures);
      const counts = new Float64Array(nClusters);

      for (let i = 0; i < nSamples; i++) {
        const c = labels[i];
        counts[c]++;
        for (let j = 0; j < nFeatures; j++) {
          newCenters[c * nFeatures + j] += X.data[i * nFeatures + j];
        }
      }

      for (let c = 0; c < nClusters; c++) {
        if (counts[c] > 0) {
          for (let j = 0; j < nFeatures; j++) {
            newCenters[c * nFeatures + j] /= counts[c];
          }
        }
      }

      centers = this.backend.array(Array.from(newCenters), [
        nClusters,
        nFeatures,
      ]);
    }

    return {
      centers,
      labels: this.backend.array(Array.from(labels), [nSamples]),
      inertia,
      nIter,
    };
  }
}
