import faust
from datetime import timedelta
from collections import deque, defaultdict
from training import train_model

# Faust App
app = faust.App(
    "training-consumer-v9",
    broker="kafka://localhost:29092",
    store="memory://training-consumer",
    table_cleanup_interval=10.0,
    consumer_auto_offset_reset="earliest"
)

# Kafka topic
schema = faust.Schema(value_serializer="json")
topic = app.topic("sensor-data", schema=schema)

WINDOW_SIZE = 50
STEP = 1
MAX_WINDOWS = 5000
MAX_ANOM_WINDOWS = 500

buffers = {}
window_pools_normal = defaultdict(list)
window_pools_anomaly = defaultdict(list)
check_types = defaultdict(bool)

@app.agent(topic)
async def training_consumer(stream):
    async for message in stream.group_by(key=lambda msg:msg["type"], name="types"):
        stype=message["type"]
        value=message["value"]
        time=message["time"]
        is_anomaly=message["anomaly"]
        if stype not in check_types:
            check_types[stype]=False

        if stype not in buffers:
            buffers[stype]=deque(maxlen=WINDOW_SIZE)
        buffers[stype].append([value,time,is_anomaly])

        if len(buffers[stype]) == WINDOW_SIZE:
            window_data = list(buffers[stype])
            if not any(m[-1] for m in window_data):
                window_pools_normal[stype].append(window_data)
                print("type: ", stype)
                print("window: ", window_data)
            else:
                window_pools_anomaly[stype].append(window_data)
            for _ in range(STEP):
                if buffers[stype]:
                    buffers[stype].popleft()
            print(stype, len(window_pools_normal[stype]), len(window_pools_anomaly[stype]))
            if len(window_pools_normal[stype]) >= MAX_WINDOWS and len(window_pools_anomaly[stype]) >= MAX_ANOM_WINDOWS and not check_types[stype]:
                print(f"{stype} için {len(window_pools_normal[stype])} pencere toplandı, training başlıyor...")
                train_model(stype, window_pools_normal[stype], window_pools_anomaly[stype])  # sensör tipine özel eğitim
                await app.stop()
                window_pools_normal[stype].clear()
                window_pools_anomaly[stype].clear()
                check_types[stype]=True
                if all(check_types.values()):
                    await app.stop()
