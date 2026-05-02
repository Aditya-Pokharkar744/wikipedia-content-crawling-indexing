import faiss
import pickle
import sys
import argparse
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os
import re
import torch
from collections import defaultdict

class QuerySearcher:
    def __init__(self, index_dir: str = "index", 
                 model_name: str = None,
                 device: str = "cuda",
                 show_progress: bool = True):
        """
        Initialize the query searcher with configurable options.
        
        Args:
            index_dir: Directory containing FAISS index and metadata
            model_name: Override model name from index metadata
            device: Computation device ('cuda' or 'cpu')
            show_progress: Show progress bars during encoding
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.show_progress = show_progress
        
        # Load index and metadata with validation
        self._load_index(index_dir)
        self._load_model(model_name)
        
        # Warmup model
        if self.device == "cuda":
            self.model.encode("warmup", convert_to_numpy=True)

    def _load_index(self, index_dir: str):
        """Load FAISS index and metadata with error checking"""
        if not os.path.exists(index_dir):
            raise FileNotFoundError(f"Index directory {index_dir} not found")
            
        # Load FAISS index
        index_path = os.path.join(index_dir, "faiss_index.bin")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        self.faiss_index = faiss.read_index(index_path)

        # Load metadata
        meta_path = os.path.join(index_dir, "metadata.pkl")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found at {meta_path}")
            
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        self.passages = self.metadata['passages']
        self.passage_to_doc = self.metadata['passage_to_doc']
        self.documents = self.metadata['documents']

    def _load_model(self, model_name: str):
        """Initialize model with compatibility checks"""
        stored_model = self.metadata.get('model_name', 'unknown')
        model_name = model_name or stored_model
        
        if model_name != stored_model:
            print(f"Warning: Using model {model_name} different from index's {stored_model}")
            
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Verify embedding dimensions match
        model_dim = self.model.get_sentence_embedding_dimension()
        index_dim = self.metadata['embedding_dim']
        if model_dim != index_dim:
            raise ValueError(f"Model dimension {model_dim} != index dimension {index_dim}")

    def _preprocess_query(self, query: str) -> str:
        """Basic query cleaning"""
        return query.strip()[:512]  # Truncate very long queries


    def search(self, query: str, k: int = 5, score_threshold: float = 0.3) -> List[Dict]:
        query = self._preprocess_query(query)
        if not query:
            return []

        # Encode query
        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True,
            show_progress_bar=self.show_progress
        ).cpu().numpy()

        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding.reshape(1, -1))

        # Search with FAISS
        search_k = min(k * 10, len(self.passages))  # Increase retrieval to get more passages
        distances, indices = self.faiss_index.search(
            query_embedding.reshape(1, -1), 
            search_k
        )

        
        # Group results by document and track max scores
        doc_scores = defaultdict(lambda: {
            'title': 0.0,
            'title_passage_idx': None,
            'content': [],
            'best_content_score': 0.0,
            'best_passage_idx': None,
            'debug_content_passages': 0  # Debug counter
        })

        # First pass: collect all relevant scores
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx < 0 or idx >= len(self.passage_to_doc):
                continue
                
            score = float(distances[0][i])
            passage_idx = idx
            doc_info = self.passage_to_doc[passage_idx]
            doc_id = doc_info['doc_id']
            
            if doc_info['is_title']:
                if score > doc_scores[doc_id]['title']:
                    doc_scores[doc_id]['title'] = score
                    doc_scores[doc_id]['title_passage_idx'] = passage_idx
            else:
                # Content passage
                doc_scores[doc_id]['content'].append((score, passage_idx))
                doc_scores[doc_id]['debug_content_passages'] += 1
        


        # Process content scores
        for doc_id, scores in doc_scores.items():
            content_scores = scores['content']
            if content_scores:
                # Sort by score, highest first
                content_scores.sort(reverse=True)
                # Take average of top 3 content matches (or fewer if less than 3)
                top_content = content_scores[:min(3, len(content_scores))]
                scores['best_content_score'] = sum(score for score, _ in top_content)/len(top_content)
                scores['best_passage_idx'] = content_scores[0][1]
            else:
                scores['best_content_score'] = 0.0
                scores['best_passage_idx'] = scores['title_passage_idx']

        # Calculate combined scores
        combined_scores = []
        for doc_id, scores in doc_scores.items():
            # Ensure we have a valid passage index
            passage_idx = scores['best_passage_idx']
            if passage_idx is None:
                passage_idx = scores['title_passage_idx']
                if passage_idx is None:
                    continue  
            
            title_weight = 0.5
            content_weight = 0.5
            # Calculate combined score - this is critical
            combined = (scores['title'] * title_weight) + (scores['best_content_score'] * content_weight)
            
            combined_scores.append((
                doc_id,
                combined,
                scores['title'],
                scores['best_content_score'],  # Store this explicitly
                passage_idx
            ))
            
        # Sort by combined score and select top k
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for item in combined_scores[:k]:
            doc_id, combined, title_score, content_score, passage_idx = item
            doc = self.documents[doc_id]
            passage = self.passages[passage_idx]
            
            results.append({
                "score": combined,
                "combined_score": combined,
                "title_score": title_score,
                "content_score": content_score,  # This should be the same value we debugged
                "passage": passage,
                "document": doc['title'],
                "url": doc['url'],
                "doc_id": doc_id
            })

        return results

    def format_result(self, result: Dict, query: str = None) -> str:
        doc_content = self.documents[result['doc_id']]['content']
        
        # Always show the first 30 words for all results
        preview_words = 30
        preview = ' '.join(doc_content.split()[:preview_words]) + "..."

        return (
            f"Score: {result['score']:.1%}\n"
            #f"Title Score: {result['title_score']:.1%}\n"
            #f"Content Score: {result['content_score']:.1%}\n"
            f"Document: {result['document']}\n"
            f"URL: {result['url']}\n"
            f"Preview: {preview}\n"
            f"{'-'*80}"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    parser.add_argument("query", type=str, help="Search query text")
    parser.add_argument("-k", type=int, default=5, help="Number of results")
    parser.add_argument("--index-dir", default="/home/sbera/Projects/IR/indexing/bert_similarity/index", help="Index directory")
    parser.add_argument("--model", default='sentence-transformers/all-mpnet-base-v2', help="Override embedding model")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--threshold", type=float, default=0.3, 
                       help="Minimum similarity score (0-1)")
    args = parser.parse_args()

    try:
        searcher = QuerySearcher(
            index_dir=args.index_dir,
            model_name=args.model,
            device="cpu" if args.cpu else "cuda"
        )
        
        results = searcher.search(args.query, k=args.k, score_threshold=args.threshold)
        
        print(f"\nFound {len(results)} results for: '{args.query}'\n")
        for i, res in enumerate(results):
            print(f"Result {i+1}:")
            # Remove the `is_top_result` argument here
            print(searcher.format_result(res, args.query))
            
    except Exception as e:
        print(f"Search failed: {str(e)}", file=sys.stderr)
        sys.exit(1)