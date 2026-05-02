import time
import json
import matplotlib.pyplot as plt
from elasticsearch import Elasticsearch, helpers
from ijson import items

def index_documents_and_measure_time(index_name, json_file_path, batch_size=500):
    es = Elasticsearch(["http://localhost:9200"], headers={"Content-Type": "application/json"})
    if not es.ping():
        raise ConnectionError("Could not connect to Elasticsearch")
    
    num_docs = []
    run_times = []
    
    with open(json_file_path, 'rb') as file:
        documents = items(file, 'item')
        batch = []
        count = 0
        start_time = time.time()
        
        for doc in documents:
            batch.append({
                "_index": index_name,
                "_source": {
                    "title": doc.get('title', ''),
                    "url": doc.get('url', ''),
                    "content": doc.get('content', '')
                }
            })
            count += 1
            
            if len(batch) >= batch_size:
                helpers.bulk(es, batch)
                batch.clear()
                elapsed_time = time.time() - start_time
                num_docs.append(count)
                run_times.append(elapsed_time)
                print(f"Indexed {count} documents in {elapsed_time:.2f} seconds")
    
    if batch:
        helpers.bulk(es, batch)
        count += len(batch)
        elapsed_time = time.time() - start_time
        num_docs.append(count)
        run_times.append(elapsed_time)
        print(f"Indexed {count} documents in {elapsed_time:.2f} seconds")
    
    return num_docs, run_times

def plot_indexing_time(num_docs, run_times):
    plt.figure(figsize=(10, 5))
    plt.plot(num_docs, run_times, marker='o', linestyle='-', color='b')
    plt.xlabel("Number of Documents")
    plt.ylabel("Indexing Time (seconds)")
    plt.title("Elasticsearch Indexing Performance")
    plt.grid(True)
    plt.savefig("indexing_runtime.png")
    plt.show()

if __name__ == "__main__":
    json_file_path = "unique_data.json"  # Replace with your JSON file path
    index_name = "documents_index"
    num_docs, run_times = index_documents_and_measure_time(index_name, json_file_path, batch_size=1000)
    plot_indexing_time(num_docs, run_times)
