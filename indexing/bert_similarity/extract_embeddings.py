import json
import os
import re
import pickle
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

class EmbeddingExtractor:
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2'):
        """
        Initialize the embedding extractor with a sentence-transformers model.
        
        Args:
            model_name (str): Pretrained sentence-transformers model name
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.tokenizer = self.model.tokenizer
        self.max_seq_length = self.model.get_max_seq_length()
        
        print(f"Using model: {model_name}")
        print(f"Max sequence length: {self.max_seq_length}")
        print(f"Device: {self.model.device}")

    def clean_text(self, text):
        """
        Clean Wikipedia text by removing markup and templates.
        
        Args:
            text (str): Raw document text
            
        Returns:
            str: Cleaned text
        """
        # Remove wiki templates and markup
        text = re.sub(r'\{\{.*?\}\}', '', text, flags=re.DOTALL)  # Templates
        text = re.sub(r'\[\[.*?\|', '', text)  # Wiki links with labels
        text = re.sub(r'\{\{|\}\}|\[\[|\]\]', '', text)  # Remaining brackets
        
        # Clean whitespace and newlines
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()

    def split_text_into_passages(self, text, min_passage_length=100):
        """
        Split cleaned text into meaningful passages based on paragraphs.
        
        Args:
            text (str): Cleaned document text
            min_passage_length (int): Minimum character length for a valid passage
            
        Returns:
            list: List of passage strings
        """
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 0]
        
        passages = []
        current_passage = []
        current_length = 0
        
        for para in paragraphs:
            para_length = len(para)
            
            # Start new passage if adding this paragraph would exceed max length
            if current_length + para_length > self.max_seq_length * 4:  # Approx token count
                if current_passage:
                    joined_passage = ' '.join(current_passage)
                    if len(joined_passage) >= min_passage_length:
                        passages.append(joined_passage)
                    current_passage = []
                    current_length = 0
                
            current_passage.append(para)
            current_length += para_length
        
        # Add remaining content
        if current_passage:
            joined_passage = ' '.join(current_passage)
            if len(joined_passage) >= min_passage_length:
                passages.append(joined_passage)
                
        return passages

    def get_embeddings(self, texts, batch_size=256):
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts (list): List of text strings
            batch_size (int): Batch size for encoding
            
        Returns:
            numpy.ndarray: Array of embeddings
        """
        return self.model.encode(texts, 
                               batch_size=batch_size,
                               show_progress_bar=True,
                               convert_to_numpy=True,
                               normalize_embeddings=True)

    def process_documents(self, input_file):
        """Process documents with duplicate title checking"""
        with open(input_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
            
        all_passages = []
        passage_to_doc = []
        seen_titles = set()
        unique_docs = []

        # Deduplication phase
        for doc in tqdm(documents, desc="Deduplicating documents"):
            clean_title = doc['title'].strip().lower()
            if clean_title in seen_titles:
                continue
            seen_titles.add(clean_title)
            unique_docs.append(doc)
        
        # Process unique documents
        for doc_idx, doc in enumerate(tqdm(unique_docs, desc="Processing documents")):
            # Add title passage
            title_passage = doc['title']
            all_passages.append(title_passage)
            passage_to_doc.append({
                'doc_id': doc_idx,
                'title': doc['title'],
                'url': doc['url'],
                'is_title': True
            })
            
            # Process content
            cleaned_text = self.clean_text(doc['content'])
            content_passages = self.split_text_into_passages(cleaned_text)
            
            for passage in content_passages:
                all_passages.append(passage)
                passage_to_doc.append({
                    'doc_id': doc_idx,
                    'title': doc['title'],
                    'url': doc['url'],
                    'is_title': False
                })
                
        return all_passages, passage_to_doc, unique_docs
                

    def extract_and_save_embeddings(self, input_file, output_file):
        """
        Main pipeline to process documents and save embeddings.
        
        Args:
            input_file (str): Path to input JSON file
            output_file (str): Path to output pickle file
        """
        # Process documents and split into passages
        passages, passage_to_doc, documents = self.process_documents(input_file)
        
        # Generate embeddings in batches
        embeddings = self.get_embeddings(passages)
        
        # Convert to float32 for FAISS compatibility
        embeddings = embeddings.astype('float32')
        
        # Save results
        with open(output_file, 'wb') as f:
            pickle.dump({
                'embeddings': embeddings,
                'passages': passages,
                'passage_to_doc': passage_to_doc,
                'documents': documents,
                'model_name': self.model_name
            }, f)
            
        print(f"Saved embeddings for {len(passages)} passages to {output_file}")

if __name__ == "__main__":
    input_file = "/scr1/sbera004/IR/Data/data.json"
    output_file = "wiki_embeddings.pkl"
    
    extractor = EmbeddingExtractor(model_name='sentence-transformers/all-mpnet-base-v2')
    extractor.extract_and_save_embeddings(input_file, output_file)