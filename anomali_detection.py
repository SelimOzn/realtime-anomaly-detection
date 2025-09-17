import faust

app = faust.App(
    "sensor_anomaly_detector",
    broker="kafka://localhost:29092",
    store="memory://",
    table_cleanup_interval=10.0
)

schema = faust.Schema(value_serializer="json")
topic = app.topic("sensor-data", schema=schema)
changelog = app.topic("sensor_window_changelog", partitions=8)
sensor_table = (app.Table("sensor_window", default=list, partitions=8, changelog_topic=changelog)
                .hopping(size=50, step=1))


@app.agent(topic)
async def processor(stream):
    async for message in stream.group_by(key = lambda msg: msg["type"], name="t_group"):
        #Producer mesajı gönderirken "key"i belirtmediğinden dolayı burada type üzerinden veriyi repartition yapmamız gerek.
        print(message)
        sensor_table[message["type"]] += [message["value"]]
        anomaly = message["anomaly"]

def is_anomaly(stype, value, anomaly):
    pass

