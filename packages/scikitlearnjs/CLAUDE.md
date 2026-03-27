# scikitlearnjs

scikit-learn for JavaScript - 1:1 API parity with Python sklearn.

## Philosophy

**This package mirrors scikit-learn's structure exactly.** If it exists in sklearn, it should exist here with the same API. If it doesn't exist in sklearn, it doesn't belong here.

## Package Structure

```
scikitlearnjs/
├── src/                    # TypeScript API (sklearn modules)
│   ├── preprocessing/      # StandardScaler, MinMaxScaler
│   ├── neural_network/     # MLPClassifier, MLPRegressor
│   ├── metrics/            # accuracy_score, mse, etc.
│   ├── model_selection/    # train_test_split
│   ├── cluster/            # KMeans
│   └── linear_model/       # LinearRegression, LogisticRegression
└── package.json            # npm package
```

## Architecture

Pure TypeScript, uses numpyjs for array operations. No Rust code here - sklearn doesn't have CNN ops.

For CNN operations (conv2d, batch_norm, pooling), use **torchjs** instead.

## Module Structure (matching sklearn)

## Quick Start

```typescript
import { createBackend } from 'numpyjs';
import {
  preprocessing,
  neural_network,
  metrics,
  trainTestSplit
} from 'scikitlearnjs';

const backend = await createBackend('wasm');

// Split data
const { XTrain, XTest, yTrain, yTest } = trainTestSplit(X, y, backend, {
  testSize: 0.2,
  randomState: 42,
});

// Scale features
const scaler = new preprocessing.StandardScaler(backend);
const XTrainScaled = await scaler.fitTransform(XTrain);
const XTestScaled = await scaler.transform(XTest);

// Train classifier
const clf = new neural_network.MLPClassifier(backend, {
  hiddenLayerSizes: [100, 50],
  activation: 'relu',
});
await clf.fit(XTrainScaled, yTrain);

// Predict and evaluate
const predictions = await clf.predict(XTestScaled);
const accuracy = metrics.accuracyScore(yTest, predictions);
console.log(`Accuracy: ${accuracy}`);
```

## Implemented Classes

### neural_network.MLPClassifier

Multi-layer Perceptron classifier:

```typescript
const clf = new neural_network.MLPClassifier(backend, {
  hiddenLayerSizes: [100, 50],
  activation: 'relu',     // 'identity' | 'logistic' | 'tanh' | 'relu'
  maxIter: 200,
  tol: 1e-4,
  alpha: 0.0001,          // L2 regularization
});

await clf.fit(X_train, y_train);
const predictions = await clf.predict(X_test);
const probabilities = await clf.predictProba(X_test);
const accuracy = await clf.score(X_test, y_test);
```

### neural_network.MLPRegressor

Multi-layer Perceptron regressor:

```typescript
const reg = new neural_network.MLPRegressor(backend, {
  hiddenLayerSizes: [100, 50],
  activation: 'relu',
});

await reg.fit(X_train, y_train);
const predictions = await reg.predict(X_test);
const r2 = await reg.score(X_test, y_test);
```

### preprocessing.StandardScaler

Standardize features (z-score normalization):

```typescript
const scaler = new preprocessing.StandardScaler(backend, {
  withMean: true,
  withStd: true,
});

const X_scaled = await scaler.fitTransform(X_train);
const X_test_scaled = await scaler.transform(X_test);
const X_original = await scaler.inverseTransform(X_scaled);
```

### preprocessing.MinMaxScaler

Scale features to a range:

```typescript
const scaler = new preprocessing.MinMaxScaler(backend, {
  featureRange: [0, 1],
});

const X_scaled = await scaler.fitTransform(X_train);
```

### linear_model.LinearRegression

Ordinary least squares regression:

```typescript
const reg = new linear_model.LinearRegression(backend, {
  fitIntercept: true,
});

await reg.fit(X_train, y_train);
const predictions = await reg.predict(X_test);
const r2 = await reg.score(X_test, y_test);
```

### linear_model.LogisticRegression

Logistic regression classifier:

```typescript
const clf = new linear_model.LogisticRegression(backend, {
  C: 1.0,              // Regularization strength (inverse)
  maxIter: 100,
  penalty: 'l2',
});

await clf.fit(X_train, y_train);
const predictions = await clf.predict(X_test);
const proba = await clf.predictProba(X_test);
```

### cluster.KMeans

K-Means clustering:

```typescript
const kmeans = new cluster.KMeans(backend, {
  nClusters: 3,
  init: 'k-means++',
  maxIter: 300,
});

await kmeans.fit(X);
const labels = kmeans.labels;
const centers = kmeans.clusterCenters;
const newLabels = await kmeans.predict(X_new);
```

### model_selection.trainTestSplit

Split data into train/test sets:

```typescript
const { XTrain, XTest, yTrain, yTest } = trainTestSplit(X, y, backend, {
  testSize: 0.2,
  randomState: 42,
  shuffle: true,
  stratify: y,  // Optional: stratified split
});
```

### metrics

Classification metrics:

```typescript
import { metrics } from 'scikitlearnjs';

const acc = metrics.accuracyScore(yTrue, yPred);
const precision = metrics.precisionScore(yTrue, yPred);
const recall = metrics.recallScore(yTrue, yPred);
const f1 = metrics.f1Score(yTrue, yPred);
const cm = metrics.confusionMatrix(yTrue, yPred, backend);
```

Regression metrics:

```typescript
const mse = metrics.meanSquaredError(yTrue, yPred);
const rmse = metrics.rootMeanSquaredError(yTrue, yPred);
const mae = metrics.meanAbsoluteError(yTrue, yPred);
const r2 = metrics.r2Score(yTrue, yPred);
```

## API Pattern

All estimators follow sklearn's API:

```typescript
// Constructor takes backend + options
const estimator = new SomeEstimator(backend, options);

// Fit to training data
await estimator.fit(X, y);

// Transform or predict
const result = await estimator.transform(X);   // transformers
const result = await estimator.predict(X);     // predictors

// Fit and transform in one step
const result = await estimator.fitTransform(X);

// Get/set parameters
estimator.getParams();
estimator.setParams({ alpha: 0.001 });

// Score
const score = await estimator.score(X_test, y_test);
```

## numpyjs Integration

All estimators use numpyjs Backend for array operations:

```typescript
import { createBackend } from 'numpyjs';

// Get backend once, pass to all estimators
const backend = await createBackend('wasm');
const scaler = new preprocessing.StandardScaler(backend);
const clf = new neural_network.MLPClassifier(backend);
```

## What Does NOT Belong Here

- Raw tensor operations (use numpyjs)
- PyTorch-style layer classes (Conv2d, LSTM) - not in sklearn public API
- Custom inference engines - use sklearn patterns
- Anything not in sklearn's actual API

## Checklist Before Committing

- [ ] `pnpm typecheck` passes
- [ ] `pnpm test` passes
- [ ] API matches sklearn 1:1
- [ ] All array ops use numpyjs backend
- [ ] New classes added to index.ts exports
