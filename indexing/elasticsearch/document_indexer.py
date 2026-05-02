from elasticsearch import Elasticsearch, helpers
import json
import os
import sys
import logging
import argparse
from typing import List, Dict, Any, Iterator
from ijson import items

class DocumentIndexer:
    def __init__(self, 
                 index_name: str, 
                 elastic_host: str = 'localhost', 
                 elastic_port: int = 9200,
                 batch_size: int = 500):
        self.index_name = index_name
        self.batch_size = batch_size
        self.es = Elasticsearch(
            [f'http://{elastic_host}:{elastic_port}'],
            headers={"Content-Type": "application/json"}
        )
        self.logger = logging.getLogger(__name__)
        if not self.es.ping():
            raise ConnectionError("Could not connect to Elasticsearch")
        self.logger.info("Successfully connected to Elasticsearch")

    def index_exists(self) -> bool:
        """Check if the index already exists"""
        return self.es.indices.exists(index=self.index_name)
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the index"""
        if not self.index_exists():
            return {"doc_count": 0}
        return self.es.indices.stats(index=self.index_name)["indices"][self.index_name]["total"]

    def create_index(self) -> None:
        """Create index if it doesn't exist"""
        if self.index_exists():
            self.logger.info(f"Index {self.index_name} already exists with {self.get_index_stats().get('docs', {}).get('count', 0)} documents")
            return
        
        mapping = {
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "refresh_interval": "30s"
                },
                "analysis": {
                    "analyzer": {
                        "custom_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stop", "snowball"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "title": {
                        "type": "text",
                        "analyzer": "custom_analyzer",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    "url": {"type": "keyword"},
                    "content": {
                        "type": "text",
                        "analyzer": "custom_analyzer"
                    },
                    "indexed_file": {"type": "keyword"}  # Track which file this document came from
                }
            }
        }
        self.es.indices.create(index=self.index_name, body=mapping)
        self.logger.info(f"Created new index: {self.index_name}")

    def document_generator(self, json_file_path: str) -> Iterator[Dict[str, Any]]:
        file_basename = os.path.basename(json_file_path)
        with open(json_file_path, 'rb') as file:
            documents = items(file, 'item')
            for i, doc in enumerate(documents):
                yield {
                    "_index": self.index_name,
                    "_id": doc.get('url', f"{file_basename}-{i}"),
                    "_source": {
                        "title": doc.get('title', ''),
                        "url": doc.get('url', ''),
                        "content": doc.get('content', ''),
                        "indexed_file": file_basename  # Track which file this came from
                    }
                }

    def is_file_indexed(self, filename: str) -> bool:
        """Check if a file has already been indexed"""
        if not self.index_exists():
            return False
            
        # Search for at least one document from this file
        query = {
            "size": 1,
            "query": {
                "term": {
                    "indexed_file": os.path.basename(filename)
                }
            }
        }
        
        result = self.es.search(index=self.index_name, body=query)
        return result["hits"]["total"]["value"] > 0

    def index_documents(self, json_file_path: str, force: bool = False) -> None:
        # Check if file is already indexed
        file_basename = os.path.basename(json_file_path)
        if not force and self.is_file_indexed(file_basename):
            self.logger.info(f"File {file_basename} is already indexed. Skipping.")
            return
            
        try:
            success_count = 0
            error_count = 0
            for ok, result in helpers.streaming_bulk(
                client=self.es,
                actions=self.document_generator(json_file_path),
                chunk_size=self.batch_size,
                raise_on_error=False,
                request_timeout=60
            ):
                if ok:
                    success_count += 1
                else:
                    error_count += 1
                    self.logger.error(f"Error indexing document: {result}")
                if (success_count + error_count) % self.batch_size == 0:
                    self.logger.info(
                        f"Indexed {success_count + error_count} documents. "
                        f"Successful: {success_count}, Failed: {error_count}"
                    )
            self.es.indices.refresh(index=self.index_name)
            self.logger.info(
                f"Indexing completed for {json_file_path}. Total: {success_count + error_count}, "
                f"Successful: {success_count}, Failed: {error_count}"
            )
        except Exception as e:
            self.logger.error(f"Error during indexing: {str(e)}")
            raise

    def search(self, query: str, size: int = 10) -> List[Dict[str, Any]]:
        search_body = {
            "size": size,
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
            "collapse": {
                "field": "title.keyword",
                "inner_hits": {
                    "name": "alternatives",
                    "size": 2
                }
            },
            "_source": ["title", "url", "content"],
            "highlight": {
                "fields": {
                    "content": {
                        "fragment_size": 200,
                        "number_of_fragments": 1,
                        "pre_tags": [""],
                        "post_tags": [""]
                    }
                }
            }
        }
        
        try:
            results = self.es.search(
                index=self.index_name,
                body=search_body
            )
            
            return [
                {
                    'title': hit['_source']['title'],
                    'url': hit['_source']['url'],
                    'content': hit.get('highlight', {}).get('content', [hit['_source']['content'][:200]])[0] + '...',
                    'score': hit['_score']
                }
                for hit in results['hits']['hits']
            ]
        except Exception as e:
            self.logger.error(f"Search error: {str(e)}")
            raise

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Elasticsearch document indexer and searcher')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index documents')
    index_parser.add_argument('directory', help='Directory containing JSON files to index')
    index_parser.add_argument('--force', action='store_true', help='Force reindexing of already indexed files')
    index_parser.add_argument('--index-name', default='documents_index', help='Name of the Elasticsearch index')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search documents')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--k', type=int, default=10, help='Number of results to return')
    search_parser.add_argument('--index-name', default='documents_index', help='Name of the Elasticsearch index')
    
    args = parser.parse_args()
    
    try:
        indexer = DocumentIndexer(args.index_name, batch_size=1000)
        
        if args.command == 'index':
            indexer.create_index()  # This will now only create if it doesn't exist
            for filename in os.listdir(args.directory):
                if filename.endswith('.json'):
                    filepath = os.path.join(args.directory, filename)
                    logging.info(f"Processing {filepath}")
                    indexer.index_documents(filepath, force=args.force)
            print(f"Indexing complete. Index '{args.index_name}' is ready for searching.")
            
        elif args.command == 'search':
            if not indexer.index_exists():
                print(f"Error: Index '{args.index_name}' does not exist. Please index documents first.")
                sys.exit(1)
                
            results = indexer.search(args.query, size=args.k)
            print(f"\nTop {args.k} results for query: '{args.query}':")
            for i, result in enumerate(results, 1):
                print(f"\nResult {i}:")
                print(f"Title: {result['title']}")
                print(f"URL: {result['url']}")
                print(f"Score: {result['score']:.2f}")
                print(f"Preview: {result['content']}")
                print("-" * 50)
        else:
            parser.print_help()
            
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()