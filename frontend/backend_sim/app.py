import json
import random
import time
from datetime import datetime

from flask import Flask, jsonify
from flask_sock import Sock

app = Flask(__name__)
sock = Sock(app)

CLASS_NAMES = [
    'openlive',
    'live',
    'message',
    'short_video',
    'video',
    'meeting',
    'phone_game',
    'cloud_game'
]


def make_distribution():
    weights = [random.random() for _ in CLASS_NAMES]
    total = sum(weights)
    return {name: round(w / total, 4) for name, w in zip(CLASS_NAMES, weights)}


def make_payload():
    dist = make_distribution()
    predicted = max(dist, key=dist.get)

    payload = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'stream': {
            'packet_size': random.randint(60, 1500),
            'iat': round(random.uniform(0.2, 180.0), 3),
            'direction': random.choice(['uplink', 'downlink'])
        },
        'metrics': {
            'accuracy': round(random.uniform(0.90, 0.98), 4),
            'recall': round(random.uniform(0.89, 0.97), 4),
            'inference_latency_ms': round(random.uniform(2.5, 18.0), 3),
            'power_w': round(random.uniform(2.2, 9.8), 3),
            'flows_per_sec': round(random.uniform(100, 280), 2)
        },
        'prediction': {
            'class_id': CLASS_NAMES.index(predicted),
            'class_name': predicted,
            'confidence': round(dist[predicted], 4)
        },
        'distribution': dist
    }
    return payload


@app.get('/api/health')
def health():
    return jsonify({'status': 'ok'})


@app.get('/api/once')
def once():
    return jsonify(make_payload())


@sock.route('/ws/telemetry')
def telemetry(ws):
    while True:
        ws.send(json.dumps(make_payload(), ensure_ascii=True))
        time.sleep(1)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
