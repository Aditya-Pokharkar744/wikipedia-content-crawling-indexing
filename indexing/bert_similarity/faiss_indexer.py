import faiss
import pickle
import os
import numpy as np
from tqdm import tqdm

class DocumentIndexer:
    def __init__(self, use_gpu=False):
        """
        Initialize the document indexer with configurable options.
        
        Args:
            use_gpu (bool): Whether to use GPU acceleration if available
        """
        self.passages = []
        self.passage_to_doc = []
        self.embeddings = None
        self.faiss_index = None
        self.documents = []
        self.use_gpu = use_gpu
        self.model_name = None

    def load_embeddings(self, embeddings_file):
        """
        Load precomputed embeddings and metadata with validation.
        
        Args:
            embeddings_file (str): Path to the embeddings file
        """
        if not os.path.exists(embeddings_file):
            raise FileNotFoundError(f"Embeddings file {embeddings_file} not found")

        with open(embeddings_file, 'rb') as f:
            data = pickle.load(f)

        # Validate loaded data
        required_keys = {'embeddings', 'passages', 'passage_to_doc', 'documents'}
        if not required_keys.issubset(data.keys()):
            missing = required_keys - data.keys()
            raise ValueError(f"Missing required keys in embeddings file: {missing}")

        self.embeddings = data['embeddings'].astype('float32')
        self.passages = data['passages']
        self.passage_to_doc = data['passage_to_doc']
        self.documents = data['documents']
        self.model_name = data.get('model_name', 'unknown')

        print(f"Loaded {len(self.documents)} documents with {len(self.passages)} passages")
        print(f"Embedding dimensions: {self.embeddings.shape[1]}")
        print(f"Model used: {self.model_name}")

    def build_index(self, index_type="IVF"):
        """
        Build optimized FAISS index with configurable types.
        
        Args:
            index_type (str): Type of index to build (IVF, Flat)
        """
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call load_embeddings() first")

        dim = self.embeddings.shape[1]
        num_vectors = self.embeddings.shape[0]

        # Configure resources
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            config = faiss.GpuClonerOptions()
            config.useFloat16 = True

        # Create appropriate index type
        if index_type == "IVF" and num_vectors > 10000:
            print("Building IVF index for large dataset")
            nlist = min(100, num_vectors // 100)  # Number of clusters
            quantizer = faiss.IndexFlatIP(dim)
            
            if self.use_gpu:
                quantizer = faiss.index_cpu_to_gpu(res, 0, quantizer)
            
            self.faiss_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            
            # Train with progress bar
            print("Training index...")
            with tqdm(total=1, desc="Index Training") as pbar:
                self.faiss_index.train(self.embeddings)
                pbar.update(1)
            
            self.faiss_index.nprobe = min(10, nlist//4)  # Balance speed/accuracy
        else:
            print("Building Flat index")
            self.faiss_index = faiss.IndexFlatIP(dim)
            if self.use_gpu:
                self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)

        # Add vectors with progress bar
        print("Adding vectors to index...")
        batch_size = 10000
        for i in tqdm(range(0, num_vectors, batch_size), desc="Indexing"):
            batch = self.embeddings[i:i+batch_size]
            self.faiss_index.add(batch)

        # Verify index health
        if self.faiss_index.ntotal != num_vectors:
            raise RuntimeError(f"Index build failed. Expected {num_vectors} vectors, got {self.faiss_index.ntotal}")

        print(f"Successfully built index with {self.faiss_index.ntotal} vectors")

    def save_index(self, output_dir, overwrite=False):
        """
        Save index with version control and validation.
        
        Args:
            output_dir (str): Directory to save index files
            overwrite (bool): Overwrite existing files
        """
        if os.path.exists(output_dir):
            existing_files = os.listdir(output_dir)
            if existing_files and not overwrite:
                raise FileExistsError(f"Output directory {output_dir} not empty. Use overwrite=True to replace")
        else:
            os.makedirs(output_dir)

        # Save FAISS index
        index_path = os.path.join(output_dir, "faiss_index.bin")
        if isinstance(self.faiss_index, faiss.GpuIndex):
            cpu_index = faiss.index_gpu_to_cpu(self.faiss_index)
            faiss.write_index(cpu_index, index_path)
        else:
            faiss.write_index(self.faiss_index, index_path)

        # Save enhanced metadata
        metadata = {
            'passages': self.passages,
            'passage_to_doc': self.passage_to_doc,
            'documents': self.documents,
            'model_name': self.model_name,
            'index_type': type(self.faiss_index).__name__,
            'embedding_dim': self.embeddings.shape[1],
            'total_vectors': self.embeddings.shape[0]
        }

        meta_path = os.path.join(output_dir, "metadata.pkl")
        with open(meta_path, 'wb') as f:
            pickle.dump(metadata, f)

        print(f"Index components saved to {output_dir}")
        print(f"- FAISS index: {os.path.getsize(index_path)//1024**2}MB")
        print(f"- Metadata: {os.path.getsize(meta_path)//1024**2}MB")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build FAISS index from embeddings")
    parser.add_argument("--input", "-i", default="wiki_embeddings.pkl", help="Input embeddings file")
    parser.add_argument("--output", "-o", default="index", help="Output directory")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")
    parser.add_argument("--overwrite", default=True, action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    try:
        indexer = DocumentIndexer(use_gpu=args.gpu)
        indexer.load_embeddings(args.input)
        indexer.build_index(index_type="IVF" if len(indexer.documents) > 10000 else "Flat")
        indexer.save_index(args.output, overwrite=args.overwrite)
    except Exception as e:
        print(f"Error building index: {str(e)}")
        exit(1)