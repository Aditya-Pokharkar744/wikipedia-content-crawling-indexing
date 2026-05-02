import ijson
import json
from typing import List, Dict

def create_sample_json(input_file: str, output_file: str, sample_size: int) -> None:
    """
    Create a sample JSON file by taking the first n items from a large JSON file.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path to save the output sample JSON file
        sample_size (int): Number of items to include in the sample
    """
    samples: List[Dict] = []
    
    # Open the file in binary mode for ijson
    with open(input_file, 'rb') as file:
        # Create an iterator for the JSON objects
        parser = ijson.items(file, 'item')
        
        # Take the first sample_size items
        for i, item in enumerate(parser):
            if i >= sample_size:
                break
            samples.append(item)
    
    # Write the samples to a new JSON file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(samples, outfile, indent=2, ensure_ascii=False)
        
if __name__ == "__main__":
    create_sample_json(
        input_file='./documents/data.json',
        output_file='./documents/sample.json',
        sample_size=100
    )