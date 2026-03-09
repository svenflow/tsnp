import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    headers: {
      // Enable SharedArrayBuffer for WASM zero-copy
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp'
    }
  },
  optimizeDeps: {
    exclude: ['@tensorflow/tfjs']
  }
});
