const puppeteer = require('puppeteer');
const http = require('http');
const fs = require('fs');
const path = require('path');

async function main() {
  const cwd = process.cwd();
  
  const server = http.createServer((req, res) => {
    let reqPath = req.url.split('?')[0];
    if (reqPath === '/') reqPath = '/zerocopy-bench.html';
    
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
      console.error('404:', req.url, '->', filePath);
      res.writeHead(404);
      res.end('Not found: ' + filePath);
    }
  });
  
  server.listen(9124);
  console.log('Server on http://localhost:9124');
  
  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();
  
  page.on('console', msg => console.log(msg.text()));
  page.on('pageerror', err => console.error('PAGE ERROR:', err.message));
  
  await page.goto('http://localhost:9124/', { waitUntil: 'networkidle0' });
  
  await page.waitForFunction('window.DONE === true', { timeout: 600000 });
  
  await browser.close();
  server.close();
}

main().catch(e => { console.error(e); process.exit(1); });
