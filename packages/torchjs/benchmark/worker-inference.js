/**
 * Worker for WebGPU inference with mapSync support
 *
 * When Chrome is started with --enable-features=WebGPUMapSyncOnWorkers,
 * buffer.mapSync() is available and MUCH faster than mapAsync().
 */

// Import torchjs (we'll inline the relevant parts for now)
let torchjs = null;
let backend = null;
let weights = null;

self.onmessage = async (e) => {
  const { type, data } = e.data;

  if (type === 'init') {
    try {
      // Import the module
      torchjs = await import('/packages/torchjs/dist/index.mjs');

      // Load weights
      const [metaRes, binRes] = await Promise.all([
        fetch('/BlazePalm/ML/converted/weights.json'),
        fetch('/BlazePalm/ML/converted/weights.bin'),
      ]);
      const metadata = await metaRes.json();
      const buffer = await binRes.arrayBuffer();
      weights = torchjs.loadWeightsFromBuffer(metadata, buffer);

      // Initialize WebGPU
      backend = await torchjs.createWebGPUBackend();
      torchjs.setBackend(backend);

      // Check if mapSync is available
      const testBuffer = backend.device?.createBuffer({
        size: 16,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      const hasMapSync = typeof testBuffer?.mapSync === 'function';
      testBuffer?.destroy();

      self.postMessage({
        type: 'ready',
        hasMapSync,
        weightCount: Object.keys(weights).length
      });
    } catch (err) {
      self.postMessage({ type: 'error', error: err.message });
    }
  }

  if (type === 'inference') {
    try {
      const testInput = {
        data: new Float32Array(data.inputBuffer),
        shape: [1, 3, 256, 256]
      };

      const t0 = performance.now();
      const output = torchjs.withBatch(() => torchjs.handLandmarks(testInput, weights));
      const t1 = performance.now();

      // Use batchGetData which will use mapSync if available
      await torchjs.WebGPUTensor.batchGetData([
        output.handflag,
        output.handedness,
        output.landmarks,
      ]);
      const t2 = performance.now();

      self.postMessage({
        type: 'result',
        inference: t1 - t0,
        getData: t2 - t1,
        total: t2 - t0,
        handflag: output.handflag.data[0],
        handedness: output.handedness.data[0],
      });
    } catch (err) {
      self.postMessage({ type: 'error', error: err.message });
    }
  }
};
