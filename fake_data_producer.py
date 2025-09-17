from kafka import KafkaProducer
import json, sys, time, random, pytz, argparse
import numpy as np
from datetime import datetime
import pytz
import math


def make_message(sensor_id, sensor_type, t, value, anomaly=False):
    h_m_s = 3600*t.hour + 60*t.minute + t.second
    unique_time = math.sin(2*math.pi*h_m_s/86400) + math.cos(2*math.pi*h_m_s/86400) # Anomaly detection modelinde gece gündüz
    # koşullarını ayırt etmek için
    return {
        "sensor_id": sensor_id,
        "type": sensor_type,
        "ts": t.isoformat(),
        "value": value,
        "anomaly": anomaly,
        "time": unique_time,
    }

def sensor_value(base, noise_scale=0.5, step=0.01):
    return base + np.random.normal(scale=noise_scale) + step

def main(brokers="localhost:9092", topic="sensor-data", sensors=5, rate=10, anomaly_prob=0.01):
    producer = KafkaProducer(
        bootstrap_servers=[brokers],
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        linger_ms=5  #Producer'ın mesajı göndermeden önce biraz beklemesi sağlanır. Böylece gelen
        #başka mesaj varsa batch halinde gönderilmesi sağlanır. Bu batch ile verimliliğin artması için
        #kafka containerda compression aktive hale getirilmeli.
    )

    sensor_types = ["temperature", "pressure", "vibration"]
    bases = [25.0, 1013.0, 0.02]

    sensor_meta = []
    for i in range(sensors):
        stype = sensor_types[i%len(sensor_types)] #Parametre ile belirlenen sensör sayısına uyum için
        #tekrar eden sensör tipleri
        base = bases[i%len(sensor_types)]
        sensor_meta.append((f"sensor_{i+1}", stype, base))

    print(f"[Producer] Starting -> brokers={brokers} topic={topic} sensors={sensors} rate={rate}hz anomaly_p={anomaly_prob}")
    i=0
    try:
        while True:
            t = datetime.now(pytz.timezone('UTC')) # Zaman karmaşıklığı olmasın diye serverda veri,
            #UTC time zone olarak tutuluyor.
            for sensor_id, stype, base in sensor_meta:
                value = sensor_value(base, noise_scale=0.2 if stype=="temperature" else 0.5)
                is_anomaly = False
                if random.random() < anomaly_prob:
                    if stype == "temperature":
                        delta = random.choice([
                            random.uniform(-15,-5),
                            random.uniform(5,15),
                        ])
                        value += delta
                    elif stype == "pressure":
                        delta = random.choice([
                            random.uniform(-80,-40),
                            random.uniform(40,80),
                        ])
                        value += delta
                    else:
                        delta = random.choice([
                            random.uniform(-1, -0.5),
                            random.uniform(0.5, 1),
                        ])
                        value += delta
                    is_anomaly = True

                msg = make_message(sensor_id, stype, t, float(value), is_anomaly)
                producer.send(topic, msg)
            i+=1
            print(f"{i}. mesaj gönderildi.")
            producer.flush() # Sync producer
            time.sleep(1/rate)
    except KeyboardInterrupt:
        print("Durduruldu")
    finally:
        producer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser() # Terminalden parametre alma
    parser.add_argument("--brokers", default="localhost:29092")
    parser.add_argument("--topic", default="sensor-data")
    parser.add_argument("--sensors", default=6, type=int)
    parser.add_argument("--rate", default=2, type=int) # Her sensör için saniyede kaç mesaj gidecek
    parser.add_argument("--anomaly_prob", default=0.02, type=float)
    args = parser.parse_args()
    main(args.brokers, args.topic, args.sensors, args.rate, args.anomaly_prob)
