#!/usr/bin/env node
// Dev server with COOP/COEP headers for SharedArrayBuffer (WASM threading)
// Supports both HTTP and HTTPS (Tailscale cert for WebGPU on mobile)
import http from 'http';
import https from 'https';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = process.argv[2] || 8099;
const HTTPS_PORT = process.argv[3] || 8443;

const MIME_TYPES = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.mjs': 'application/javascript',
  '.wasm': 'application/wasm',
  '.json': 'application/json',
  '.css': 'text/css',
};

function handler(req, res) {
  const proto = req.socket.encrypted ? 'https' : 'http';
  const urlPath = new URL(req.url, `${proto}://localhost:${PORT}`).pathname;
  let filePath = path.join(__dirname, urlPath === '/' ? 'bench.html' : urlPath);

  // Security: prevent directory traversal
  const resolved = path.resolve(filePath);
  if (!resolved.startsWith(path.resolve(__dirname))) {
    res.writeHead(403);
    res.end('Forbidden');
    return;
  }

  const ext = path.extname(filePath);
  const contentType = MIME_TYPES[ext] || 'application/octet-stream';

  fs.readFile(filePath, (err, content) => {
    if (err) {
      if (err.code === 'ENOENT') {
        res.writeHead(404);
        res.end(`Not found: ${req.url}`);
      } else {
        res.writeHead(500);
        res.end(`Server error: ${err.message}`);
      }
      return;
    }

    res.writeHead(200, {
      'Content-Type': contentType,
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Resource-Policy': 'cross-origin',
      'Cache-Control': 'no-cache',
    });
    res.end(content);
  });
}

// HTTP server (localhost, desktop)
const httpServer = http.createServer(handler);
httpServer.listen(PORT, '0.0.0.0', () => {
  console.log(`HTTP:  http://localhost:${PORT}/`);
});

// HTTPS server (Tailscale, needed for WebGPU on mobile)
const certFile = path.join(__dirname, 'sven.tail6669f2.ts.net.crt');
const keyFile = path.join(__dirname, 'sven.tail6669f2.ts.net.key');

if (fs.existsSync(certFile) && fs.existsSync(keyFile)) {
  const httpsServer = https.createServer({
    cert: fs.readFileSync(certFile),
    key: fs.readFileSync(keyFile),
  }, handler);
  httpsServer.listen(HTTPS_PORT, '0.0.0.0', () => {
    console.log(`HTTPS: https://sven.tail6669f2.ts.net:${HTTPS_PORT}/`);
    console.log(`\nMobile: use HTTPS URL for WebGPU support`);
  });
} else {
  console.log(`No TLS certs found — HTTPS disabled. Run: tailscale cert sven.tail6669f2.ts.net`);
}
