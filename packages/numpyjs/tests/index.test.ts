/**
 * JS backend test suite (runs in Node)
 *
 * Tests are parameterized across backends. WebGPU tests run separately
 * in browser mode via webgpu.test.ts.
 * WASM backend tests live in packages/numpyjs-wasm.
 */

import { describe, beforeAll } from 'vitest';
import { Backend } from './test-utils';
import { creationTests } from './creation.test';
import { mathTests } from './math.test';
import { linalgTests } from './linalg.test';
import { statsTests } from './stats.test';
import { randomTests } from './random.test';
import { manipulationTests } from './manipulation.test';
import { phase2Tests } from './phase2.test';
import { coverageTests } from './coverage.test';
import { numpyApiTests } from './numpy-api.test';
import { dtypeScalarTests } from './dtype-scalar.test';
import { parityTests } from './parity.test';
import { coverageBoostTests } from './coverage-boost.test';
import { coverageBoost2Tests } from './coverage-boost-2.test';
import { coverageBoost3Tests } from './coverage-boost-3.test';
import { coverageBoost4Tests } from './coverage-boost-4.test';

import { createJsBackend } from '../src/js-backend';

describe('numpyjs js backend', () => {
  let backend: Backend;

  beforeAll(() => {
    backend = createJsBackend();
  });

  const getBackend = () => backend;

  creationTests(getBackend);
  mathTests(getBackend);
  linalgTests(getBackend);
  statsTests(getBackend);
  randomTests(getBackend);
  manipulationTests(getBackend);
  phase2Tests(getBackend);
  coverageTests(getBackend);
  numpyApiTests(getBackend);
  dtypeScalarTests(getBackend);
  parityTests(getBackend);
  coverageBoostTests(getBackend);
  coverageBoost2Tests(getBackend);
  coverageBoost3Tests(getBackend);
  coverageBoost4Tests(getBackend);
});
