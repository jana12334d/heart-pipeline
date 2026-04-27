from kafka import KafkaProducer
import json
import time
import csv

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

with open('../data/heart.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        producer.send('heart_data', value=row)
        print("Sent:", row)
        time.sleep(1)
