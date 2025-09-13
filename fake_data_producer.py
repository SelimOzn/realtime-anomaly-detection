from kafka import KafkaProducer
import json
import time
import random

def produce_message():
    producer = KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    i = 0
    while True:
        message = {
            "temperature": random.uniform(20,30),
            "pressure": random.uniform(1,2),
            "speed": random.uniform(50,100),
        }

        producer.send("sensor-data", message)
        i+=1
        print(f"{i}. mesaj gönderildi.")
        time.sleep(1)

if __name__ == '__main__':
    produce_message()