# Scrapy settings for wikipedia_scraper project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://docs.scrapy.org/en/latest/topics/settings.html
#     https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://docs.scrapy.org/en/latest/topics/spider-middleware.html

BOT_NAME = "wikipedia_scraper"

SPIDER_MODULES = ["wikipedia_scraper.spiders"]
NEWSPIDER_MODULE = "wikipedia_scraper.spiders"


# Crawl responsibly by identifying yourself (and your website) on the user-agent
#USER_AGENT = "wikipedia_scraper (+http://www.yourdomain.com)"

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# Configure maximum concurrent requests performed by Scrapy (default: 16)
#CONCURRENT_REQUESTS = 32

# Configure a delay for requests for the same website (default: 0)
# See https://docs.scrapy.org/en/latest/topics/settings.html#download-delay
# See also autothrottle settings and docs
#DOWNLOAD_DELAY = 3
# The download delay setting will honor only one of:
#CONCURRENT_REQUESTS_PER_DOMAIN = 16
#CONCURRENT_REQUESTS_PER_IP = 16

# Disable cookies (enabled by default)
#COOKIES_ENABLED = False

# Disable Telnet Console (enabled by default)
#TELNETCONSOLE_ENABLED = False

# Override the default request headers:
#DEFAULT_REQUEST_HEADERS = {
#    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
#    "Accept-Language": "en",
#}

# Enable or disable spider middlewares
# See https://docs.scrapy.org/en/latest/topics/spider-middleware.html
#SPIDER_MIDDLEWARES = {
#    "wikipedia_scraper.middlewares.WikipediaScraperSpiderMiddleware": 543,
#}

# Enable or disable downloader middlewares
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#DOWNLOADER_MIDDLEWARES = {
#    "wikipedia_scraper.middlewares.WikipediaScraperDownloaderMiddleware": 543,
#}

# Enable or disable extensions
# See https://docs.scrapy.org/en/latest/topics/extensions.html
#EXTENSIONS = {
#    "scrapy.extensions.telnet.TelnetConsole": None,
#}

# Configure item pipelines
# See https://docs.scrapy.org/en/latest/topics/item-pipeline.html
#ITEM_PIPELINES = {
#    "wikipedia_scraper.pipelines.WikipediaScraperPipeline": 300,
#}

# Enable and configure the AutoThrottle extension (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/autothrottle.html
#AUTOTHROTTLE_ENABLED = True
# The initial download delay
#AUTOTHROTTLE_START_DELAY = 5
# The maximum download delay to be set in case of high latencies
#AUTOTHROTTLE_MAX_DELAY = 60
# The average number of requests Scrapy should be sending in parallel to
# each remote server
#AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
# Enable showing throttling stats for every response received:
#AUTOTHROTTLE_DEBUG = False

# Enable and configure HTTP caching (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html#httpcache-middleware-settings
#HTTPCACHE_ENABLED = True
#HTTPCACHE_EXPIRATION_SECS = 0
#HTTPCACHE_DIR = "httpcache"
#HTTPCACHE_IGNORE_HTTP_CODES = []
#HTTPCACHE_STORAGE = "scrapy.extensions.httpcache.FilesystemCacheStorage"

# Set settings whose default value is deprecated to a future-proof value
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
FEED_EXPORT_ENCODING = "utf-8"
# settings.py

# Set the User-Agent to simulate a real browser
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

# Set a high concurrency to increase the crawl speed
CONCURRENT_REQUESTS = 100  # Number of concurrent requests
CONCURRENT_REQUESTS_PER_DOMAIN = 100  # Number of requests to a single domain
CONCURRENT_REQUESTS_PER_IP = 100  # Number of requests to a single IP

# Disable the download delay to maximize the crawl speed (use with caution)
DOWNLOAD_DELAY = 0  # 0 means no delay between requests

# Enable auto-throttling (Scrapy will dynamically adjust download speed)
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1  # The initial delay before starting a crawl
AUTOTHROTTLE_MAX_DELAY = 10  # The maximum delay in case of high latency
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0  # Target concurrency per remote server

# Disable cookies to avoid unnecessary overhead
COOKIES_ENABLED = False

# Set download timeout (increase if you encounter slow responses)
DOWNLOAD_TIMEOUT = 15

# Log everything to monitor progress (optional)
LOG_LEVEL = 'INFO'

# Stop the crawl if it reaches the max pages
CLOSESPIDER_PAGECOUNT = 500000  # Stop the spider after crawling 500,000 pages
# settings.py

# Set the format for exporting data (choose 'jsonlines' for line-by-line JSON objects)
FEED_FORMAT = 'jsonlines'  # Scrapy will output one JSON object per line
FEED_URI = 'output.json'   # Specify the output file location
