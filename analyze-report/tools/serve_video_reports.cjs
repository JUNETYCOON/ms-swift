const fs = require('fs');
const http = require('http');
const path = require('path');

const root = path.resolve(process.argv[2] || '.');
const port = Number.parseInt(process.argv[3] || '18767', 10);
const host = '127.0.0.1';

const mimeTypes = {
  '.css': 'text/css; charset=utf-8',
  '.csv': 'text/csv; charset=utf-8',
  '.gif': 'image/gif',
  '.htm': 'text/html; charset=utf-8',
  '.html': 'text/html; charset=utf-8',
  '.jpeg': 'image/jpeg',
  '.jpg': 'image/jpeg',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.jsonl': 'application/x-ndjson; charset=utf-8',
  '.m4v': 'video/x-m4v',
  '.mkv': 'video/x-matroska',
  '.mov': 'video/quicktime',
  '.mp4': 'video/mp4',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.webm': 'video/webm',
};

function sendText(response, statusCode, body) {
  const payload = Buffer.from(body, 'utf8');
  response.writeHead(statusCode, {
    'Content-Type': 'text/plain; charset=utf-8',
    'Content-Length': payload.length,
    'Cache-Control': 'no-store',
  });
  response.end(payload);
}

function resolveRequestPath(requestUrl) {
  const requestPath = new URL(requestUrl, `http://${host}:${port}`).pathname;
  const decoded = decodeURIComponent(requestPath === '/' ? '/local-videos.html' : requestPath);
  const candidate = path.resolve(root, `.${decoded.split('/').join(path.sep)}`);
  const rootKey = root.toLocaleLowerCase();
  const candidateKey = candidate.toLocaleLowerCase();
  if (candidateKey !== rootKey && !candidateKey.startsWith(`${rootKey}${path.sep}`)) return null;
  return candidate;
}

function parseRange(header, size) {
  const match = /^bytes=(\d*)-(\d*)$/.exec(header || '');
  if (!match) return null;
  let start;
  let end;
  if (match[1] === '' && match[2] !== '') {
    const suffixLength = Number.parseInt(match[2], 10);
    if (!Number.isFinite(suffixLength) || suffixLength <= 0) return null;
    start = Math.max(0, size - suffixLength);
    end = size - 1;
  } else {
    start = Number.parseInt(match[1], 10);
    end = match[2] === '' ? size - 1 : Number.parseInt(match[2], 10);
  }
  if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || start >= size || end < start) return null;
  return { start, end: Math.min(end, size - 1) };
}

function serveFile(request, response, filePath, stat) {
  const contentType = mimeTypes[path.extname(filePath).toLowerCase()] || 'application/octet-stream';
  const commonHeaders = {
    'Content-Type': contentType,
    'Accept-Ranges': 'bytes',
    'Cache-Control': 'no-cache',
    'X-Content-Type-Options': 'nosniff',
  };
  const rangeHeader = request.headers.range;
  if (rangeHeader) {
    const range = parseRange(rangeHeader, stat.size);
    if (!range) {
      response.writeHead(416, { ...commonHeaders, 'Content-Range': `bytes */${stat.size}` });
      response.end();
      return;
    }
    const contentLength = range.end - range.start + 1;
    response.writeHead(206, {
      ...commonHeaders,
      'Content-Length': contentLength,
      'Content-Range': `bytes ${range.start}-${range.end}/${stat.size}`,
    });
    if (request.method === 'HEAD') {
      response.end();
      return;
    }
    fs.createReadStream(filePath, { start: range.start, end: range.end }).pipe(response);
    return;
  }

  response.writeHead(200, { ...commonHeaders, 'Content-Length': stat.size });
  if (request.method === 'HEAD') {
    response.end();
    return;
  }
  fs.createReadStream(filePath).pipe(response);
}

if (!fs.existsSync(root) || !fs.statSync(root).isDirectory()) {
  throw new Error(`报告根目录不存在：${root}`);
}

const server = http.createServer((request, response) => {
  if (request.method !== 'GET' && request.method !== 'HEAD') {
    sendText(response, 405, 'Method Not Allowed');
    return;
  }
  let filePath;
  try {
    filePath = resolveRequestPath(request.url);
  } catch {
    sendText(response, 400, 'Bad Request');
    return;
  }
  if (!filePath) {
    sendText(response, 403, 'Forbidden');
    return;
  }
  fs.stat(filePath, (error, stat) => {
    if (error || !stat.isFile()) {
      sendText(response, 404, 'Not Found');
      return;
    }
    serveFile(request, response, filePath, stat);
  });
});

server.on('clientError', (_error, socket) => {
  socket.end('HTTP/1.1 400 Bad Request\r\n\r\n');
});

server.listen(port, host, () => {
  process.stdout.write(`Serving ${root} at http://${host}:${port}/\n`);
});
