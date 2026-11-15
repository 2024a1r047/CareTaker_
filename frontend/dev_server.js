/**
 * Frontend dev server — same engine and behaviour as `serve .` (clean URLs: /dashboard, /profile,
 * "/" opens index.html), with one fix:
 *
 *   `serve` redirects profile.html?recipient_id=57  ->  /profile   and DROPS "?recipient_id=57",
 *   so pages opened from a link lost the patient. Here the same redirect keeps the query string.
 *
 * Start:  npm run dev      (port 3000; if it is busy, the next free port)
 */
const http = require('http');
const os = require('os');
const path = require('path');
const handler = require('serve-handler');

const ROOT = __dirname;
const CONFIG = { public: ROOT, cleanUrls: true, directoryListing: false };

// Backend (FastAPI). Pages opened on another address (e.g. http://192.168.29.4:3000) call "/api/...":
// pass those through to the backend instead of answering with an HTML page.
const BACKEND_HOST = process.env.BACKEND_HOST || '127.0.0.1';
const BACKEND_PORT = Number(process.env.BACKEND_PORT) || 8000;
const isBackendPath = url => /^\/(api|socket\.io)(\/|\?|$)/.test(url);

function proxyHttp(req, res) {
    const up = http.request({ host: BACKEND_HOST, port: BACKEND_PORT, method: req.method, path: req.url,
                              headers: { ...req.headers, host: `${BACKEND_HOST}:${BACKEND_PORT}` } }, upRes => {
        res.writeHead(upRes.statusCode, upRes.headers);
        upRes.pipe(res);
    });
    up.on('error', () => {
        if (!res.headersSent) res.writeHead(502, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ detail: 'Backend not reachable on port ' + BACKEND_PORT + ' — is it running?' }));
    });
    req.pipe(up);
}

// Addresses of this computer (to recognise a request coming from the laptop itself)
const OWN_IPS = new Set(['127.0.0.1', '::1']);
for (const list of Object.values(os.networkInterfaces())) for (const a of list || []) OWN_IPS.add(a.address);
const fromThisComputer = req => OWN_IPS.has(String(req.socket.remoteAddress || '').replace(/^::ffff:/, ''));
const isLocalHost = h => /^(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$/i.test(h || '');

const server = http.createServer((req, res) => {
    if (isBackendPath(req.url)) return proxyHttp(req, res);
    // On this laptop, always use http://localhost:PORT. The browser keeps a separate login per address,
    // and blocks camera / microphone on a plain-http network address (192.168.x.x) - so opening the
    // network link here made pages lose the login and the camera. Other devices are not redirected.
    if (req.method === 'GET' && fromThisComputer(req) && !isLocalHost(req.headers.host)) {
        const port = server.address() && server.address().port;
        res.writeHead(302, { Location: `http://localhost:${port}${req.url}` });
        return res.end();
    }
    const q = req.url.indexOf('?');
    const pathname = q >= 0 ? req.url.slice(0, q) : req.url;
    const search = q >= 0 ? req.url.slice(q) : '';
    // the clean-URL redirect, done here so the "?..." part is kept
    if (search && /(\.html|\/index)$/.test(pathname)) {
        const clean = pathname.replace(/(\.html|\/index)$/, '') || '/';
        res.writeHead(301, { Location: clean + search });
        return res.end();
    }
    return handler(req, res, CONFIG);
});

// WebSockets (voice chat, live updates) under /api or /socket.io: pass the upgrade through too
server.on('upgrade', (req, socket, head) => {
    if (!isBackendPath(req.url)) return socket.destroy();
    const net = require('net');
    const up = net.connect(BACKEND_PORT, BACKEND_HOST, () => {
        const lines = [`${req.method} ${req.url} HTTP/${req.httpVersion}`];
        for (let i = 0; i < req.rawHeaders.length; i += 2) {
            const k = req.rawHeaders[i];
            lines.push(`${k}: ${k.toLowerCase() === 'host' ? BACKEND_HOST + ':' + BACKEND_PORT : req.rawHeaders[i + 1]}`);
        }
        up.write(lines.join('\r\n') + '\r\n\r\n');
        if (head && head.length) up.write(head);
        up.pipe(socket); socket.pipe(up);
    });
    up.on('error', () => socket.destroy());
    socket.on('error', () => up.destroy());
});

function addresses(port) {
    const out = [`http://localhost:${port}`];
    for (const list of Object.values(os.networkInterfaces())) {
        for (const a of list || []) {
            if (a.family === 'IPv4' && !a.internal) { out.push(`http://${a.address}:${port}`); break; }
        }
        if (out.length > 1) break;
    }
    return out;
}

function listen(port, triesLeft) {
    server.once('error', err => {
        if (err.code === 'EADDRINUSE' && triesLeft > 0) {
            console.log(`Port ${port} is busy, trying ${port + 1}...`);
            listen(port + 1, triesLeft - 1);
        } else {
            console.error(err);
            process.exit(1);
        }
    });
    server.listen(port, () => {
        const [local, network] = addresses(port);
        console.log('\n   Serving CareTaker frontend (' + path.basename(ROOT) + ')\n');
        console.log('   - Local:    ' + local + '   <- open this one on this laptop');
        if (network) console.log('   - Network:  ' + network + '   (for other devices; on this laptop it opens localhost)');
        console.log('');
    });
}

listen(Number(process.env.PORT) || 3000, 10);
