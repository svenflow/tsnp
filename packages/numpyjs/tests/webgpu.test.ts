/**
 * WebGPU backend test suite
 *
 * Runs the shared test suite against the WebGPU backend.
 * Must run in browser mode (vitest-browser + playwright).
 */

import { describe, beforeAll, it } from 'vitest';
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
import { webgpuCoverageTests } from './webgpu-coverage.test';
import { initWebGPUBackend, createWebGPUBackend } from '../src/webgpu-backend';

describe('numpyjs webgpu', () => {
  let backend: Backend;

  beforeAll(async () => {
    await initWebGPUBackend();
    backend = createWebGPUBackend();
  });

  it('webgpu backend available', () => {
    if (!backend) throw new Error('WebGPU backend not initialized');
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
  webgpuCoverageTests(getBackend);
});
