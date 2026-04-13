import { getApiBaseUrl } from './api';

function buildWsUrl(token) {
  const rawUrl = import.meta.env.VITE_WS_URL;
  const baseUrl = rawUrl ? new URL(rawUrl) : new URL(getApiBaseUrl());

  baseUrl.protocol = baseUrl.protocol === 'https:' ? 'wss:' : 'ws:';
  if (!rawUrl) {
    baseUrl.pathname = '/ws/telemetry';
  }
  if (token) {
    baseUrl.searchParams.set('token', token);
  }

  return baseUrl.toString();
}

export function createTelemetrySocket({ token, onMessage, onStatus, onUnauthorized }) {
  const wsUrl = buildWsUrl(token);

  let ws = null;
  let reconnectTimer = null;
  let shouldReconnect = true;

  function connect() {
    onStatus?.('connecting');
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      onStatus?.('connected');
    };

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        onMessage?.(payload);
      } catch (error) {
        onStatus?.('error');
      }
    };

    ws.onerror = () => {
      onStatus?.('error');
    };

    ws.onclose = (event) => {
      onStatus?.('disconnected');
      if (event.code === 4401 || event.code === 4403 || event.code === 1008) {
        shouldReconnect = false;
        onUnauthorized?.();
        return;
      }
      if (!shouldReconnect) {
        return;
      }
      reconnectTimer = setTimeout(connect, 1200);
    };
  }

  function close() {
    shouldReconnect = false;
    clearTimeout(reconnectTimer);
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      ws.close();
    }
  }

  connect();
  return { close };
}
