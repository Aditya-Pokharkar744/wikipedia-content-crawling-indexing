from flask import Flask, request, render_template, jsonify
import os
import sys
from indexing.elasticsearch.document_indexer import DocumentIndexer
from indexing.bert_similarity.query_index import QuerySearcher
import requests


BERT_SERVER_URL = "http://localhost:5001"

app = Flask(__name__)

# Initialize searchers (with lazy loading to avoid startup overhead)
es_indexer = None
bert_searcher = None

def get_es_indexer():
    global es_indexer
    if es_indexer is None:
        try:
            es_indexer = DocumentIndexer(index_name="documents_index")
        except Exception as e:
            return None, str(e)
    return es_indexer, None

def get_bert_searcher():
    global bert_searcher
    if bert_searcher is None:
        try:
            bert_searcher = QuerySearcher(
                index_dir="index",
                device="cuda" if os.environ.get("USE_GPU", "true").lower() == "true" else "cpu"
            )
        except Exception as e:
            return None, str(e)
    return bert_searcher, None

@app.route('/')
def home():
    return render_template('search.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query', '')
    search_type = request.form.get('search_type', 'elasticsearch')
    results_count = int(request.form.get('k', 10))
    
    if not query:
        return jsonify({
            'error': 'No query provided',
            'results': []
        })
    
    results = []
    error = None
    
    if search_type == 'elasticsearch':
        indexer, err = get_es_indexer()
        if err:
            error = f"Elasticsearch error: {err}"
        else:
            try:
                if not indexer.index_exists():
                    error = "Elasticsearch index does not exist. Please index documents first."
                else:
                    # Get full content from Elasticsearch without truncation
                    es_results = indexer.search(query, size=results_count)
                    
                    # Override the search function to get the full content
                    search_body = {
                        "size": results_count,
                        "query": {
                            "bool": {
                                "should": [
                                    {
                                        "multi_match": {
                                            "query": query,
                                            "fields": ["title^3", "content"],
                                            "type": "best_fields",
                                            "fuzziness": "AUTO"
                                        }
                                    },
                                    {
                                        "match_phrase": {
                                            "title": {
                                                "query": query,
                                                "boost": 2
                                            }
                                        }
                                    }
                                ]
                            }
                        },
                        "_source": ["title", "url", "content"],
                    }
                    
                    try:
                        full_results = indexer.es.search(
                            index=indexer.index_name,
                            body=search_body
                        )
                        
                        results = [
                            {
                                'title': hit['_source']['title'],
                                'url': hit['_source']['url'],
                                'content': hit['_source']['content'],
                                'score': hit['_score']
                            }
                            for hit in full_results['hits']['hits']
                        ]
                    except Exception:
                        # Fallback to original results if the custom search fails
                        results = es_results
            except Exception as e:
                error = f"Elasticsearch search error: {str(e)}"
    else:  # BERT similarity
        try:
            response = requests.post(
                f"{BERT_SERVER_URL}/search",
                json={
                    "query": query,
                    "k": results_count,
                    "threshold": 0.3
                }
            )
            
            if response.status_code != 200:
                error = f"BERT server error: {response.json().get('error', 'Unknown error')}"
            else:
                bert_results = response.json()["results"]
                results = [
                    {
                        'title': r['document'],
                        'url': r['url'],
                        'content': r['passage'],
                        'score': r['score']
                    }
                    for r in bert_results
                ]
        except requests.exceptions.RequestException as e:
            error = f"BERT server connection error: {str(e)}"
    
    return jsonify({
        'error': error,
        'results': results,
        'search_type': search_type
    })

@app.route('/status')
def status():
    """Check the status of both search engines"""
    es_status = "Not initialized"
    bert_status = "Not initialized"
    
    # Check Elasticsearch
    indexer, err = get_es_indexer()
    if err:
        es_status = f"Error: {err}"
    elif indexer:
        try:
            if indexer.index_exists():
                stats = indexer.get_index_stats()
                doc_count = stats.get('docs', {}).get('count', 0)
                es_status = f"Connected, {doc_count} documents indexed"
            else:
                es_status = "Connected, no index exists"
        except Exception as e:
            es_status = f"Error: {str(e)}"
    
    # Check BERT searcher
    searcher, err = get_bert_searcher()
    if err:
        bert_status = f"Error: {err}"
    elif searcher:
        try:
            model_name = searcher.metadata.get('sentence-transformers/all-mpnet-base-v2', 'unknown')
            passage_count = len(searcher.passages)
            bert_status = f"Loaded, using model {model_name}, {passage_count} passages indexed"
        except Exception as e:
            bert_status = f"Error: {str(e)}"
    
    return jsonify({
        'elasticsearch': es_status,
        'bert': bert_status
    })

if __name__ == "__main__":
    # Create necessary imports for the modules
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Import your actual modules (assuming they're in the same directory or accessible via PYTHONPATH)
    from indexing.elasticsearch.document_indexer import DocumentIndexer
    from indexing.bert_similarity.query_index import QuerySearcher
    
    app.run(debug=True, host='0.0.0.0', port=5000)
