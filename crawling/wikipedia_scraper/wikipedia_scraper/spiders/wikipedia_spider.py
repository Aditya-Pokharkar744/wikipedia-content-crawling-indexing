import scrapy
from scrapy.crawler import CrawlerProcess  # Explicitly import CrawlerProcess
from tqdm import tqdm
import re
from urllib.parse import unquote
import argparse
import os

class WikipediaSpider(scrapy.Spider):
    name = "wikipedia"
    allowed_domains = ['wikipedia.org']

    def __init__(self, seed_file, max_pages, hops_away, output_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Load URLs from external file
        self.start_urls = self.load_start_urls(seed_file)
        
        self.page_count = 0
        self.max_pages = int(max_pages)
        self.hops_away = int(hops_away)
        self.output_dir = output_dir
        self.progress_bar = tqdm(total=self.max_pages, desc="Pages Scraped", unit="pages")

        # Pre-compile regex patterns
        self.citation_pattern = re.compile(r'\[\d+\]|\[citation needed\]', re.IGNORECASE)
        self.whitespace_pattern = re.compile(r'\s+')

        # Track visited URLs to avoid duplicates
        self.visited_urls = set()

    @staticmethod
    def load_start_urls(file_path):
        """Load start URLs from a text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                urls = [line.strip() for line in f if line.strip()]
            return urls
        except FileNotFoundError:
            print(f"Error: {file_path} not found.")
            return []

    def parse(self, response):
        """Parse a Wikipedia article page."""
        if self.page_count >= self.max_pages:
            self.crawler.engine.close_spider(self, 'page_limit_reached')
            return
        
        # Skip non-article pages
        if any(x in response.url for x in [
            '/Special:', '/Talk:', '/User:', '/Help:', 
            '/File:', '/Wikipedia:', '/Template:', '/Category:'
        ]):
            return
        
        # Skip already visited URLs
        if response.url in self.visited_urls:
            return
        self.visited_urls.add(response.url)

        # Extract title from the page or fallback to the URL
        title = response.css('h1#firstHeading::text').get()
        if not title or title == "Unknown Title":  # Fallback to extracting title from URL
            print(f"Selector failed for URL: {response.url}")
            title = self.extract_title_from_url(response.url)
            print(f"Title extracted from URL: {title}")
        
        # Get main content
        paragraphs = response.css('div.mw-parser-output > p::text, div.mw-parser-output > p > *::text').getall()
        if paragraphs:
            # Clean content
            content = ' '.join(paragraphs)
            content = self.citation_pattern.sub('', content)
            content = self.whitespace_pattern.sub(' ', content).strip()
            
            if content:  # Only yield if we have actual content
                self.page_count += 1
                self.progress_bar.update(1)
                
                yield {
                    'title': title,
                    'url': response.url,
                    'content': content
                }
        
        # Follow links to other Wikipedia articles
        if response.meta.get('hops', 0) < self.hops_away:
            for link in response.css('a::attr(href)').getall():
                if link.startswith('/wiki/') and not any(x in link for x in [
                    '/Special:', '/Talk:', '/User:', '/Help:', 
                    '/File:', '/Wikipedia:', '/Template:', '/Category:'
                ]):
                    full_url = response.urljoin(link)
                    if full_url not in self.visited_urls:  # Only follow if not visited
                        yield response.follow(link, self.parse, meta={'hops': response.meta.get('hops', 0) + 1})

    def closed(self, reason):
        """Clean up resources when spider is closed."""
        self.progress_bar.close()

    @staticmethod
    def extract_title_from_url(url):
        """
        Extract the title from a Wikipedia URL.
        Example: https://en.wikipedia.org/wiki/Page_Title -> "Page Title"
        """
        # Extract the part after '/wiki/'
        if '/wiki/' in url:
            title_part = url.split('/wiki/')[-1]
            # URL decode to handle special characters
            title = unquote(title_part)
            # Replace underscores with spaces
            title = title.replace('_', ' ')
            return title
        else:
            return "Unknown Title"

def main():
    parser = argparse.ArgumentParser(description="Wikipedia Crawler")
    parser.add_argument("seed_file", type=str, help="Path to the seed file containing starting URLs")
    parser.add_argument("num_pages", type=int, help="Maximum number of pages to crawl")
    parser.add_argument("hops_away", type=int, help="Maximum number of hops away from the seed URLs")
    parser.add_argument("output_dir", type=str, help="Directory to save the output JSON file")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Run the spider
    process = CrawlerProcess(settings={
        "FEEDS": {
            os.path.join(args.output_dir, "output.json"): {"format": "json"},
        },
    })

    process.crawl(WikipediaSpider, seed_file=args.seed_file, max_pages=args.num_pages, hops_away=args.hops_away, output_dir=args.output_dir)
    process.start()

if __name__ == "__main__":
    main()