import faust

app = faust.App(
    "inference_consumer",
    broker="kafka://localhost:29092",
    store="memory://inference_consumer",
    table_cleanup_interval=10,
)
schema = faust.Schema(value_serializer="json")
topic = app.topic("sensor-data", schema=schema)

@app.agent(topic)
async def read_loop(stream):
    vib_model =
    async for message in stream:
