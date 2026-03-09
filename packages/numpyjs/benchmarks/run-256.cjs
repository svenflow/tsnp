const puppeteer = require('puppeteer');
const http = require('http');
const fs = require('fs');
const path = require('path');

async function main() {
  const cwd = process.cwd();
  
  const html = `<!DOCTYPE html>
<html><head><meta charset="UTF-8"></head><body><pre id="log"></pre>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/tf-backend-wasm.min.js"></script>
<script type="module">
const L = document.getElementById('log');
const log = m => { L.textContent += m + '\\n'; console.log(m); };

const NT = 8;
tf.wasm.setThreadsCount(NT);
await tf.setBackend('wasm'); await tf.ready();

const r = await import('./pkg/rumpy_wasm.js');
await r.default();
await r.initThreadPool(NT);
log('tfjs + rumpy ready (8t each)\\n');

function med(fn, n=20) {
  for(let i=0;i<10;i++) fn();
  const t=[]; for(let i=0;i<n;i++){const s=performance.now();fn();t.push(performance.now()-s);}
  t.sort((a,b)=>a-b); return t[n>>1];
}
function maxDiff(a,b) { let m=0; for(let i=0;i<a.length;i++){const d=Math.abs(a[i]-b[i]); if(d>m)m=d;} return m; }

const getMem = () => r.wasmMemory().buffer;

log('256x256 benchmark (20 iterations, more warmup):');
log('');

for (let run = 1; run <= 5; run++) {
  const n = 256;
  const Adata = new Float32Array(n*n).map(Math.random);
  const Bdata = new Float32Array(n*n).map(Math.random);

  const tA = tf.tensor2d(Adata,[n,n]), tB = tf.tensor2d(Bdata,[n,n]);
  const tf_t = med(() => { const c=tf.matMul(tA,tB); c.dataSync(); c.dispose(); }, 20);
  const ref = (()=>{const c=tf.matMul(tA,tB);const d=c.dataSync();c.dispose();return d;})();
  tA.dispose(); tB.dispose();

  const bufA = r.allocF32(n*n);
  const bufB = r.allocF32(n*n);
  const bufC = r.allocF32(n*n);
  const bufPB = r.allocF32(r.packedBSize(n, n));

  new Float32Array(getMem(), bufA.ptr(), n*n).set(Adata);
  new Float32Array(getMem(), bufB.ptr(), n*n).set(Bdata);
  r.packBInPlace(bufB, bufPB, n, n);

  const zcPre_t = med(() => r.matmulF32PrepackedZeroCopy(bufA, bufPB, bufC, n, n, n), 20);
  const zcDyn_t = med(() => r.matmulF32ZeroCopy(bufA, bufB, bufC, n, n, n), 20);

  const outView = new Float32Array(getMem(), bufC.ptr(), n*n);
  const err = maxDiff(outView, ref);
  const ok = err < 1e-2 ? '✅' : '❌';

  const best = Math.min(zcPre_t, zcDyn_t);
  const which = zcPre_t <= zcDyn_t ? 'pre' : 'dyn';
  const ratio = best / tf_t;
  const marker = ratio <= 1.0 ? '⭐' : '';

  log('Run ' + run + ': tfjs=' + tf_t.toFixed(3) + 'ms, zc-pre=' + zcPre_t.toFixed(3) + 'ms, zc-dyn=' + zcDyn_t.toFixed(3) + 'ms → best=' + best.toFixed(3) + 'ms (' + which + ') ' + ratio.toFixed(2) + 'x ' + marker + ' ' + ok);

  bufA.free(); bufB.free(); bufC.free(); bufPB.free();
}

window.DONE = true;
</script></body></html>`;

  const server = http.createServer((req, res) => {
    let reqPath = req.url.split('?')[0];
    if (reqPath === '/') {
      res.writeHead(200, { 
        'Content-Type': 'text/html',
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp'
      });
      res.end(html);
      return;
    }
    
    let filePath = path.join(cwd, reqPath);
    
    const stat = fs.statSync(filePath, { throwIfNoEntry: false });
    if (stat && stat.isDirectory()) {
      const pkgJs = path.join(filePath, 'rumpy_wasm.js');
      if (fs.existsSync(pkgJs)) {
        filePath = pkgJs;
      }
    }
    
    const ext = path.extname(filePath);
    const contentType = {
      '.html': 'text/html',
      '.js': 'application/javascript',
      '.wasm': 'application/wasm',
      '.json': 'application/json'
    }[ext] || 'text/plain';
    
    try {
      const content = fs.readFileSync(filePath);
      res.writeHead(200, { 
        'Content-Type': contentType,
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp'
      });
      res.end(content);
    } catch (e) {
      res.writeHead(404);
      res.end('Not found');
    }
  });
  
  server.listen(9125);
  
  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();
  
  page.on('console', msg => console.log(msg.text()));
  page.on('pageerror', err => console.error('PAGE ERROR:', err.message));
  
  await page.goto('http://localhost:9125/', { waitUntil: 'networkidle0' });
  
  await page.waitForFunction('window.DONE === true', { timeout: 120000 });
  
  await browser.close();
  server.close();
}

main().catch(e => { console.error(e); process.exit(1); });
