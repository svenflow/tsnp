/**
 * WASM Backend test suite
 *
 * Runs the shared numpyjs test suite against the WASM backend.
 * Tests are imported from numpyjs and parameterized with createWasmBackend.
 */

import { describe, beforeAll, it } from 'vitest';
import type { Backend } from 'numpyjs';
import { createWasmBackend } from '../src/index.js';

// Import shared test modules from numpyjs
import { creationTests } from '../../numpyjs/tests/creation.test';
import { mathTests } from '../../numpyjs/tests/math.test';
import { linalgTests } from '../../numpyjs/tests/linalg.test';
import { statsTests } from '../../numpyjs/tests/stats.test';
import { randomTests } from '../../numpyjs/tests/random.test';
import { manipulationTests } from '../../numpyjs/tests/manipulation.test';
import { phase2Tests } from '../../numpyjs/tests/phase2.test';

describe('numpyjs-wasm', () => {
  let backend: Backend;

  beforeAll(async () => {
    backend = await createWasmBackend();
  });

  it('wasm backend available', () => {
    if (!backend) throw new Error('WASM backend not initialized');
  });

  const getBackend = () => backend;

  creationTests(getBackend);
  mathTests(getBackend);
  linalgTests(getBackend);
  statsTests(getBackend);
  randomTests(getBackend);
  manipulationTests(getBackend);
  phase2Tests(getBackend);
});
