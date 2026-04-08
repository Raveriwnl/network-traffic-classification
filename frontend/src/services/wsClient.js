export function createTelemetrySocket({ onMessage, onStatus }) {
  const wsUrl =
    import.meta.env.VITE_WS_URL ||
    'ws://127.0.0.1:5000/ws/telemetry';

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

    ws.onclose = () => {
      onStatus?.('disconnected');
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
