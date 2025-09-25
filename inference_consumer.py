from collections import deque
import faust
import numpy as np
import torch
from load_model import load_model
from faust.web import View
from prometheus_client import CollectorRegistry, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = faust.App(
    "inference_consumer-v4",
    broker="kafka://localhost:29092",
    store="memory://inference_consumer",
    table_cleanup_interval=10.0,
    consumer_auto_offset_reset="latest",
    web_port=8000,
    web_enable_metrics=True,
)

WINDOW_SIZE = 50
schema = faust.Schema(value_serializer="json")
topic = app.topic("sensor-data", schema=schema)
alerts_topic = app.topic("alerts", value_type=dict)
windows = {}

@app.agent(topic)
async def read_loop(stream):
    vib_model, vib_dev, vib_mean, vib_std, vib_th, vib_input_dim = load_model("vibration")
    pres_model, pres_dev, pres_mean, pres_std, pres_th, pres_input_dim = load_model("pressure")
    temp_model, temp_dev, temp_mean, temp_std, temp_th, temp_input_dim = load_model("temperature")

    async for message in stream.group_by(key=lambda msg:msg["type"], name="types"):
        stype = message["type"]
        if stype not in windows:
            windows[stype] = deque(maxlen=WINDOW_SIZE)
        window = windows[stype]
        window.append([message["value"], message["time"], message["anomaly"]])
        if len(window) == WINDOW_SIZE:
            if stype == "pressure":
                model, dev, mean, std, th, input_dim = pres_model, pres_dev, pres_mean, pres_std, pres_th, pres_input_dim
            elif stype == "temperature":
                model, dev, mean, std, th, input_dim = temp_model, temp_dev, temp_mean, temp_std, temp_th, temp_input_dim
            elif stype == "vibration":
                model, dev, mean, std, th, input_dim = vib_model, vib_dev, vib_mean, vib_std, vib_th, vib_input_dim
            else:
                await alert_missing_model(stype, message["timestamp"])
                continue

            X_window = np.array(window).astype(np.float32).reshape(1, WINDOW_SIZE, 3)
            X_sensor = X_window[:, :, 0:1]
            X_time = X_window[:, :, 1:2]
            X_labels = X_window[:, :, 2]
            X_sensor_norm = (X_sensor-mean.reshape(1,1,1)) / std.reshape(1,1,1)

            X_window = np.concatenate((X_sensor_norm, X_time), axis=2)
            X_tensor = torch.from_numpy(X_window).float().to(torch.device(dev))
            with torch.no_grad():
                out = model(X_tensor)
                mse = ((X_tensor[:, :, 0] - out[:, :, 0]) ** 2).squeeze(0)

            #labels_pred = (1 if mse > th else 0)
            labels_pred =  np.array((mse > th))
            labels_true = np.array(X_labels)
            acc = np.sum(labels_pred == labels_true) / WINDOW_SIZE
            window_accuracy.set(acc)
            window.popleft()
        windows[stype] = window

async def alert_missing_model(sensor_type, time):
    alert_msg = {
        "event": "MODEL_MISSING",
        "sensor_type": sensor_type,
        "severity": "warning",
        "timestamp": time
    }
    await alerts_topic.send(value=alert_msg)


registry = CollectorRegistry()
window_accuracy = Gauge("window_accuracy", "Accuracy per window", registry=registry)

@app.page('/metrics')
class MetricsView(View):
    async def get(self, request):
        data = generate_latest(registry)
        return self.bytes(
            data,
            headers={"Content-Type": CONTENT_TYPE_LATEST}
        )