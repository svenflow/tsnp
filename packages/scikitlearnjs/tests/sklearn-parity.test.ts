/**
 * scikitlearnjs parity tests
 *
 * These tests are ported from sklearn's actual test suite to verify
 * we match sklearn behavior exactly, including edge cases.
 *
 * Source: sklearn/preprocessing/tests/test_data.py
 *         sklearn/linear_model/tests/test_base.py
 *         sklearn/metrics/tests/test_classification.py
 *         sklearn/cluster/tests/test_k_means.py
 *
 * To regenerate expected values:
 *   uv run python3 -c "from sklearn... import ..."
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { preprocessing, linear_model, metrics, cluster } from '../src/index.js';
import { createTestBackend, toNDArray, to1DArray, expectArraysClose } from './test-utils.js';
import type { Backend } from 'numpyjs';

let backend: Backend;

beforeAll(() => {
  backend = createTestBackend();
});

// ============================================================================
// StandardScaler tests - ported from sklearn/preprocessing/tests/test_data.py
// ============================================================================

describe('preprocessing.StandardScaler', () => {
  // Ported from test_standard_scaler_1d
  describe('test_standard_scaler_1d cases', () => {
    it('single row: mean equals X, scale is ones', async () => {
      // sklearn: single row should set mean to the values, scale to 1
      // Python:
      // X = np.array([[1., 2., 3., 4., 5.]])
      // scaler.fit(X)
      // mean_ -> [1.0, 2.0, 3.0, 4.0, 5.0]
      // scale_ -> [1.0, 1.0, 1.0, 1.0, 1.0]
      // transform(X) -> [[0.0, 0.0, 0.0, 0.0, 0.0]]

      const X = toNDArray([[1.0, 2.0, 3.0, 4.0, 5.0]]);
      const scaler = new preprocessing.StandardScaler(backend);
      await scaler.fit(X);

      expectArraysClose(scaler.mean.data, [1.0, 2.0, 3.0, 4.0, 5.0]);
      expectArraysClose(scaler.scale.data, [1.0, 1.0, 1.0, 1.0, 1.0]);

      const transformed = await scaler.transform(X);
      expectArraysClose(transformed.data, [0.0, 0.0, 0.0, 0.0, 0.0]);
    });

    it('single column: computes mean and std of column', async () => {
      // Python:
      // X = np.array([[1.], [2.], [3.], [4.], [5.]])
      // mean_ -> [3.0]
      // scale_ -> [1.4142135623730951]
      // transformed -> [[-1.414...], [-0.707...], [0.0], [0.707...], [1.414...]]

      const X = toNDArray([[1.0], [2.0], [3.0], [4.0], [5.0]]);
      const scaler = new preprocessing.StandardScaler(backend);
      await scaler.fit(X);

      expectArraysClose(scaler.mean.data, [3.0]);
      expectArraysClose(scaler.scale.data, [1.4142135623730951]);

      const transformed = await scaler.transform(X);
      const expected = [
        -1.414213562373095,
        -0.7071067811865475,
        0.0,
        0.7071067811865475,
        1.414213562373095,
      ];
      expectArraysClose(transformed.data, expected);
    });

    it('constant feature: scale is 1, transformed is zeros', async () => {
      // sklearn handles constant features by setting scale=1
      // Python:
      // X = np.array([[5.], [5.], [5.], [5.]])
      // mean_ -> [5.0]
      // scale_ -> [1.0]  # Not 0!
      // transformed -> [[0.0], [0.0], [0.0], [0.0]]

      const X = toNDArray([[5.0], [5.0], [5.0], [5.0]]);
      const scaler = new preprocessing.StandardScaler(backend);
      await scaler.fit(X);

      expectArraysClose(scaler.mean.data, [5.0]);
      expectArraysClose(scaler.scale.data, [1.0]); // Important: sklearn uses 1.0 not 0.0

      const transformed = await scaler.transform(X);
      expectArraysClose(transformed.data, [0.0, 0.0, 0.0, 0.0]);
    });

    it('standard 2D case', async () => {
      // Python:
      // X = np.array([[0., 0.], [0., 0.], [1., 1.], [1., 1.]])
      // mean_ -> [0.5, 0.5]
      // scale_ -> [0.5, 0.5]
      // transformed -> [[-1, -1], [-1, -1], [1, 1], [1, 1]]

      const X = toNDArray([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]]);
      const scaler = new preprocessing.StandardScaler(backend);
      await scaler.fit(X);

      expectArraysClose(scaler.mean.data, [0.5, 0.5]);
      expectArraysClose(scaler.scale.data, [0.5, 0.5]);

      const transformed = await scaler.transform(X);
      const expected = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0];
      expectArraysClose(transformed.data, expected);
    });

    it('inverse_transform recovers original data', async () => {
      const X = toNDArray([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]]);
      const scaler = new preprocessing.StandardScaler(backend);
      await scaler.fit(X);

      const transformed = await scaler.transform(X);
      const inverse = await scaler.inverseTransform(transformed);

      const originalFlat = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
      expectArraysClose(inverse.data, originalFlat);
    });
  });
});

// ============================================================================
// MinMaxScaler tests - ported from sklearn test cases
// ============================================================================

describe('preprocessing.MinMaxScaler', () => {
  it('basic scaling to [0, 1]', async () => {
    // Python:
    // X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    // transformed: [[0.0, 0.0], [0.333..., 0.333...], [0.666..., 0.666...], [1.0, 1.0]]

    const X = toNDArray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);
    const scaler = new preprocessing.MinMaxScaler(backend);
    await scaler.fit(X);
    const transformed = await scaler.transform(X);

    expectArraysClose(scaler.dataMin.data, [1.0, 2.0]);
    expectArraysClose(scaler.dataMax.data, [7.0, 8.0]);

    const expected = [
      0.0, 0.0,
      0.33333333333333337, 0.3333333333333333,
      0.6666666666666666, 0.6666666666666667,
      1.0, 1.0,
    ];
    expectArraysClose(transformed.data, expected, 5);
  });
});

// ============================================================================
// Metrics tests - ported from sklearn/metrics/tests/test_classification.py
// ============================================================================

describe('metrics (classification)', () => {
  // sklearn test values
  const yTrueArr = [0, 0, 1, 1, 0, 1, 0, 1];
  const yPredArr = [0, 1, 1, 1, 0, 0, 0, 1];

  it('accuracy_score', () => {
    // Python: accuracy_score(y_true, y_pred) -> 0.75
    const yTrue = to1DArray(yTrueArr);
    const yPred = to1DArray(yPredArr);
    expect(metrics.accuracyScore(yTrue, yPred)).toBeCloseTo(0.75, 6);
  });

  it('precision_score', () => {
    // Python: precision_score(y_true, y_pred) -> 0.75
    const yTrue = to1DArray(yTrueArr);
    const yPred = to1DArray(yPredArr);
    expect(metrics.precisionScore(yTrue, yPred)).toBeCloseTo(0.75, 6);
  });

  it('recall_score', () => {
    // Python: recall_score(y_true, y_pred) -> 0.75
    const yTrue = to1DArray(yTrueArr);
    const yPred = to1DArray(yPredArr);
    expect(metrics.recallScore(yTrue, yPred)).toBeCloseTo(0.75, 6);
  });

  it('f1_score', () => {
    // Python: f1_score(y_true, y_pred) -> 0.75
    const yTrue = to1DArray(yTrueArr);
    const yPred = to1DArray(yPredArr);
    expect(metrics.f1Score(yTrue, yPred)).toBeCloseTo(0.75, 6);
  });

  it('perfect predictions', () => {
    const y = to1DArray([0, 1, 0, 1]);
    expect(metrics.accuracyScore(y, y)).toBe(1.0);
    expect(metrics.precisionScore(y, y)).toBe(1.0);
    expect(metrics.recallScore(y, y)).toBe(1.0);
    expect(metrics.f1Score(y, y)).toBe(1.0);
  });

  it('all wrong predictions', () => {
    const yTrue2 = to1DArray([0, 0, 0, 0]);
    const yPred2 = to1DArray([1, 1, 1, 1]);
    expect(metrics.accuracyScore(yTrue2, yPred2)).toBe(0.0);
  });
});

describe('metrics (regression)', () => {
  // sklearn test values
  const yTrueArr = [3.0, -0.5, 2.0, 7.0];
  const yPredArr = [2.5, 0.0, 2.0, 8.0];

  it('mean_squared_error', () => {
    // Python: mean_squared_error(y_true, y_pred) -> 0.375
    const yTrue = to1DArray(yTrueArr);
    const yPred = to1DArray(yPredArr);
    expect(metrics.meanSquaredError(yTrue, yPred)).toBeCloseTo(0.375, 6);
  });

  it('mean_absolute_error', () => {
    // Python: mean_absolute_error(y_true, y_pred) -> 0.5
    const yTrue = to1DArray(yTrueArr);
    const yPred = to1DArray(yPredArr);
    expect(metrics.meanAbsoluteError(yTrue, yPred)).toBeCloseTo(0.5, 6);
  });

  it('r2_score', () => {
    // Python: r2_score(y_true, y_pred) -> 0.9486081370449679
    const yTrue = to1DArray(yTrueArr);
    const yPred = to1DArray(yPredArr);
    expect(metrics.r2Score(yTrue, yPred)).toBeCloseTo(0.9486081370449679, 6);
  });

  it('perfect predictions r2', () => {
    const y = to1DArray([1.0, 2.0, 3.0]);
    expect(metrics.r2Score(y, y)).toBe(1.0);
  });
});

// ============================================================================
// KMeans tests - ported from sklearn/cluster/tests/test_k_means.py
// ============================================================================

describe('cluster.KMeans', () => {
  it('basic clustering structure', async () => {
    // KMeans is stochastic, verify structure not exact values
    const X = toNDArray([
      [1, 2], [1.5, 1.8], [5, 8], [8, 8],
      [1, 0.6], [9, 11], [2, 2], [2.5, 2.5],
    ]);

    const kmeans = new cluster.KMeans(backend, { nClusters: 2, randomState: 42 });
    await kmeans.fit(X);

    // Should have 2 cluster centers with 2 features each
    expect(kmeans.clusterCenters.shape).toEqual([2, 2]);

    // Labels should be 0 or 1
    expect(kmeans.labels.data.length).toBe(8);
    for (let i = 0; i < kmeans.labels.data.length; i++) {
      expect(kmeans.labels.data[i]).toBeGreaterThanOrEqual(0);
      expect(kmeans.labels.data[i]).toBeLessThan(2);
    }

    // Inertia should be positive
    expect(kmeans.inertia).toBeGreaterThan(0);
  });
});

// ============================================================================
// LinearRegression tests - ported from sklearn/linear_model/tests/test_base.py
// ============================================================================

describe('linear_model.LinearRegression', () => {
  it('simple linear fit', async () => {
    // Python:
    // X = np.array([[1], [2], [3], [4], [5]])
    // y = np.array([2.1, 4.0, 5.9, 8.1, 9.9])
    // coef_ -> [1.97]
    // intercept_ -> 0.09
    // predictions -> [2.06, 4.03, 6.0, 7.97, 9.94]

    const X = toNDArray([[1], [2], [3], [4], [5]]);
    const y = to1DArray([2.1, 4.0, 5.9, 8.1, 9.9]);

    const lr = new linear_model.LinearRegression(backend);
    await lr.fit(X, y);

    expectArraysClose(lr.coef.data, [1.97], 2);
    expect(lr.intercept).toBeCloseTo(0.09, 1);

    const predictions = await lr.predict(X);
    const expected = [2.06, 4.03, 6.0, 7.97, 9.94];
    expectArraysClose(predictions.data, expected, 1);
  });
});
