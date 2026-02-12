import threading
import requests
import time

URL = "http://localhost:5000/api/forecast"
STORES = [1, 5, 10, 20, 45]

def make_request(store_id):
    start = time.time()
    try:
        resp = requests.get(f"{URL}?store={store_id}", timeout=30)
        print(f"Store {store_id}: Status {resp.status_code}, Time: {time.time() - start:.2f}s")
    except Exception as e:
        print(f"Store {store_id}: Error: {e}")

threads = []
print("Starting Concurrency Stress Test (Parallel AI Training)...")
for s in STORES:
    t = threading.Thread(target=make_request, args=(s,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
print("Stress Test Complete.")
