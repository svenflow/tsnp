# scikitlearnjs Design Plan

**NOTE: This file was previously named NNJS-DESIGN-PLAN.md. The approach has changed to follow sklearn 1:1 parity.**

## Goal

1:1 API parity with Python's scikit-learn

## Architecture (Matches sklearn)

```
scikitlearnjs/
├── preprocessing/           # sklearn.preprocessing
│   ├── StandardScaler      # StandardScaler().fit_transform(X)
│   ├── MinMaxScaler        # MinMaxScaler().fit_transform(X)
│   ├── LabelEncoder        # LabelEncoder().fit_transform(y)
│   └── OneHotEncoder       # OneHotEncoder().fit_transform(X)
├── neural_network/          # sklearn.neural_network
│   ├── MLPClassifier       # Multi-layer Perceptron classifier
│   └── MLPRegressor        # Multi-layer Perceptron regressor
├── metrics/                 # sklearn.metrics
│   ├── accuracy_score
│   ├── mean_squared_error
│   ├── mean_absolute_error
│   └── r2_score
├── model_selection/         # sklearn.model_selection
│   ├── train_test_split
│   └── cross_val_score
├── cluster/                 # sklearn.cluster
│   ├── KMeans
│   └── DBSCAN
├── linear_model/            # sklearn.linear_model
│   ├── LinearRegression
│   └── LogisticRegression
└── ensemble/                # sklearn.ensemble
    ├── RandomForestClassifier
    └── RandomForestRegressor
```

## API Pattern

All estimators follow sklearn's fit/predict pattern:

```typescript
import { preprocessing, neural_network, metrics } from 'scikitlearnjs';
import { createBackend } from 'numpyjs';

const backend = await createBackend('wasm');

// Preprocessing
const scaler = new preprocessing.StandardScaler(backend);
const X_scaled = scaler.fitTransform(X_train);
const X_test_scaled = scaler.transform(X_test);

// Neural Network
const clf = new neural_network.MLPClassifier(backend, {
  hiddenLayerSizes: [100, 50],
  activation: 'relu',
  maxIter: 500,
});
clf.fit(X_scaled, y_train);
const y_pred = clf.predict(X_test_scaled);

// Metrics
const accuracy = metrics.accuracyScore(y_test, y_pred);
```

## Implementation Priority

For MediaPipe inference (FaceMesh, HandPose), we need:

### Phase 1: Core Infrastructure
1. Base Estimator interface (fit, predict, score)
2. Backend integration with numpyjs

### Phase 2: Preprocessing
1. StandardScaler - input normalization
2. MinMaxScaler

### Phase 3: Neural Networks
1. MLPRegressor - for loading pre-trained MediaPipe models
2. Internal layers (Conv2d, Dense, etc.) - NOT exported, only used internally by MLPRegressor

### Phase 4: Metrics
1. mean_squared_error
2. accuracy_score

### Phase 5: Model Selection
1. train_test_split

## Neural Network Internals

The `neural_network` module exposes sklearn's API (MLPClassifier, MLPRegressor) but internally uses layer implementations. These layers are private implementation details:

```typescript
// INTERNAL - not exported (sklearn doesn't expose these)
class _Conv2d { forward(x: NDArray): NDArray }
class _Dense { forward(x: NDArray): NDArray }
class _BatchNorm { forward(x: NDArray): NDArray }

// PUBLIC - exported (matches sklearn API)
class MLPClassifier {
  fit(X: NDArray, y: NDArray): this
  predict(X: NDArray): NDArray
  predict_proba(X: NDArray): NDArray
  score(X: NDArray, y: NDArray): number
}

class MLPRegressor {
  fit(X: NDArray, y: NDArray): this
  predict(X: NDArray): NDArray
  score(X: NDArray, y: NDArray): number
}
```

## MediaPipe Support

For MediaPipe models, we'll provide a loader that creates pre-configured MLPRegressor:

```typescript
// Extension for loading pre-trained models (not in sklearn, but useful)
import { loadMediaPipeModel } from 'scikitlearnjs/neural_network';

const model = await loadMediaPipeModel(backend, 'facemesh');
const landmarks = model.predict(imageData);  // Returns 468x3 landmarks
```

## What Belongs Where

### numpyjs (numpy parity):
- zeros, ones, arange, linspace
- sin, cos, exp, log, sqrt
- matmul, dot, transpose
- sum, mean, std, min, max
- reshape, flatten, concatenate
- random, randn, randint

### scikitlearnjs (sklearn parity):
- StandardScaler, MinMaxScaler, LabelEncoder
- MLPClassifier, MLPRegressor
- accuracy_score, mean_squared_error
- train_test_split
- KMeans, DBSCAN
- LinearRegression, LogisticRegression
- RandomForestClassifier, RandomForestRegressor

### NOT in either (doesn't exist in numpy/sklearn):
- Conv2d as public API (internal only)
- Session pattern (we just use backend directly)
- DAG execution (internal to MLPRegressor)

## Success Criteria

1. **API parity**: sklearn Python code can be ported to JS with minimal changes
2. **MediaPipe working**: FaceMesh and HandPose produce correct landmarks via MLPRegressor
3. **Tests passing**: Unit tests for all sklearn-equivalent functionality
