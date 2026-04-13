let unauthorizedHandler = null;
let authTokenGetter = () => null;

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');

export function getApiBaseUrl() {
  return API_BASE_URL;
}

export function setUnauthorizedHandler(handler) {
  unauthorizedHandler = handler;
}

export function setAuthTokenGetter(getter) {
  authTokenGetter = getter;
}

function buildUrl(path) {
  if (/^https?:\/\//.test(path)) {
    return path;
  }
  return `${API_BASE_URL}${path}`;
}

export async function apiFetch(path, options = {}) {
  const { skipAuth = false, body, headers = {}, ...rest } = options;
  const requestHeaders = new Headers(headers);
  const token = skipAuth ? null : authTokenGetter();

  if (body != null && !requestHeaders.has('Content-Type')) {
    requestHeaders.set('Content-Type', 'application/json');
  }

  if (token) {
    requestHeaders.set('Authorization', `Bearer ${token}`);
  }

  const response = await fetch(buildUrl(path), {
    ...rest,
    headers: requestHeaders,
    body
  });

  const contentType = response.headers.get('content-type') || '';
  const payload = contentType.includes('application/json')
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    if (response.status === 401) {
      unauthorizedHandler?.();
    }

    const message = typeof payload === 'string'
      ? payload
      : payload?.detail || 'Request failed.';

    throw new Error(message);
  }

  return payload;
}