import faust
from datetime import timedelta
from collections import deque, defaultdict
from training import train_model

# Faust App
app = faust.App(
    "training-consumer-v3",
    broker="kafka://localhost:29092",
    store="memory://training-consumer",
    table_cleanup_interval=10.0,
    consumer_auto_offset_reset="earliest"
)

# Kafka topic
schema = faust.Schema(value_serializer="json")
topic = app.topic("sensor-data", schema=schema)

WINDOW_SIZE = 50
STEP = 5
MAX_WINDOWS = 5000

buffers = {}
window_pools = defaultdict(list)

@app.agent(topic)
async def training_consumer(stream):
    async for message in stream.group_by(key=lambda msg:msg["type"], name="types"):
        stype=message["type"]
        value=message["value"]
        time=message["time"]

        if not message["anomaly"]:
            if stype not in buffers:
                buffers[stype]=deque(maxlen=WINDOW_SIZE)
            buffers[stype].append([value,time])

            if len(buffers[stype]) == WINDOW_SIZE:
                window_data = list(buffers[stype])
                window_pools[stype].append(window_data)

                for _ in range(STEP):
                    if buffers[stype]:
                        buffers[stype].popleft()

                if len(window_pools[stype]) >= MAX_WINDOWS:
                    print(f"{stype} için {len(window_pools[stype])} pencere toplandı, training başlıyor...")
                    train_model(stype, window_pools[stype])  # sensör tipine özel eğitim
                    window_pools[stype].clear()