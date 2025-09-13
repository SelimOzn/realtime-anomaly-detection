from kafka import KafkaProducer
import json
import time
import random

producer = KafkaProducer(
    bootstrap_servers="localhost:29092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

def generate_message():
    # %90 normal veri, %10 anomali
    if random.random() < 0.9:
        return {
            "temperature": random.uniform(20, 30),
            "pressure": random.uniform(1, 2),
            "speed": random.uniform(50, 100),
            "anomaly": False
        }
    else:
        return {
            "temperature": random.choice([random.uniform(-20, 0), random.uniform(80, 120)]),
            "pressure": random.choice([random.uniform(-1, 0.5), random.uniform(5, 15)]),
            "speed": random.choice([random.uniform(-100, 0), random.uniform(300, 1000)]),
            "anomaly": True
        }

if __name__ == "__main__":
    i=0
    while True:
        message = generate_message()
        producer.send("sensor-data", message)
        i+=1
        print(f"{i}. mesaj gönderildi: ", message)
        time.sleep(1)