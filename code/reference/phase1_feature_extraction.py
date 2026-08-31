import json
import time 
import pandas as pd
import urllib.parse
from urllib.parse import urlparse
import tldextract
import whois
import datetime
import requests
import logging 
import shutil
import concurrent.futures
import ipaddress
import re
import socket
import idna
import ipaddress
from pathlib import Path
from pyspark.sql import SparkSession 
from pyspark.sql.functions import col, pandas_udf, when, monotonically_increasing_id, PandasUDFType, from_json
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
from cachetools import TTLCache
from dateutil import parser 
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.ERROR,  # Only log errors
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logging.disable(logging.CRITICAL)

spark = SparkSession.builder \
    .appName("URLFeatureExtraction") \
    .config("spark.sql.shuffle.partitions", "100") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.cores", "4") \
    .config("spark.default.parallelism", "200") \
    .getOrCreate()

# Define TRANCO constants 
TRANCO_CSV_FILE = "tranco_list.csv"
TRANCO_URL = "https://tranco-list.eu/download/G63WK/full"

# domain_age_cache = {}
domain_age_cache = TTLCache(maxsize=10000, ttl=86400)  # Store up to 10k domains for 24 hours

def is_valid_ip(ip):
    """Check if a given string is a valid IPv4 or IPv6 address."""
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False

def get_domain_ages_bulk(domains):
    """Fetch WHOIS data for multiple domains using concurrent requests."""
    domain_ages = {}

    def normalize_datetime(dt):
        """Ensure all datetimes are in UTC and offset-aware."""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=datetime.timezone.utc)  # Assume UTC if missing
        return dt.astimezone(datetime.timezone.utc)  # Convert to UTC

    def clean_date_string(date_str):
        """Remove unwanted timezone formats, weekdays, and brackets before parsing."""
        date_str = re.sub(r"\s*\(GMT[+-]\d+:\d+\)", "", date_str)  # Remove GMT offset
        date_str = re.sub(r"[\[\]]", "", date_str)  # Remove square brackets
        date_str = re.sub(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+", "", date_str, flags=re.IGNORECASE)  # Remove weekday
        return date_str.strip()

    def fetch_whois(domain):
        retries = 3
        for i in range(retries):
            try:
                result = whois.whois(domain)
                creation_date = None

                # Try standard WHOIS fields first
                standard_fields = ["creation_date", "registered_on", "registration_date", "reg_date", "domain_created", "[Registered Date]"]
                for field in standard_fields:
                    if hasattr(result, field):
                        value = getattr(result, field)
                        if isinstance(value, list):  # Get earliest valid date
                            value = next((date for date in value if isinstance(date, datetime.datetime)), value[0])
                        if isinstance(value, str):
                            value = parser.parse(clean_date_string(value), fuzzy=True)  # Clean before parsing
                        if isinstance(value, datetime.datetime):
                            creation_date = normalize_datetime(value)
                            break  # Stop if we found a valid date

                # Extract from raw text dynamically (fallback)
                if not creation_date and hasattr(result, "text"):
                    date_pattern = re.compile(
                        r"(?:(?:\[[^\]]*\]|Created on|Domain created|assigned|Registered On|registered on|created\s*date|creation\s*date|created|registered|domain\s+created|created\s+on)\s*[:\-]?\s*)"
                        r"(\w{3}\s+\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\d{4}|\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}\s+\w+\s+\d{4}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4}\s*\d{2}:\d{2}:\d{2})",
                        re.IGNORECASE
                    )
                    matches = date_pattern.findall(result.text)
                    for match in matches:
                        try:
                            cleaned_match = clean_date_string(match)  # Clean before parsing
                            parsed_date = parser.parse(cleaned_match, fuzzy=True)
                            if parsed_date:
                                creation_date = normalize_datetime(parsed_date)
                                break  # Use the first valid date found
                        except Exception:
                            continue

                # Compute domain age if found
                if isinstance(creation_date, datetime.datetime):
                    now = datetime.datetime.now(datetime.timezone.utc)
                    age_in_days = (now - creation_date).days
                    return domain, age_in_days

            except Exception as e:
                logging.error(f"WHOIS error for {domain} (attempt {i+1}): {e}")
                time.sleep(5)

        return domain, -1  # Return -1 if WHOIS fails

    # Use ThreadPoolExecutor for parallel execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(fetch_whois, domains)

    domain_ages.update(results)
    return domain_ages

def update_tranco_list():
    """Use existing Tranco list without updating."""
    csv_path = Path(TRANCO_CSV_FILE)

    if not csv_path.exists():
        logging.error("Tranco list file not found and updating is disabled.")
        return

    logging.info("Using existing Tranco list without updating.")

def preload_tranco_list():
    """Preloads the Tranco list into memory as a dictionary."""
    update_tranco_list()
    try:
        tranco_df = pd.read_csv(TRANCO_CSV_FILE, header=None, names=["domain"])
        return {domain: rank + 1 for rank, domain in enumerate(tranco_df["domain"])}
    except Exception as e:
        logging.error(f"Error loading Tranco list: {e}")
        return {}

def get_web_traffic(url, tranco_domains):
    """Calculates the Dowdall score for a URL's Tranco rank."""
    try:
        domain = tldextract.extract(url).registered_domain
        if not domain:
            return 0

        rank = tranco_domains.get(domain)
        return 1 / rank if rank else 0
    except Exception as e:
        logging.warning(f"Error checking Tranco rank for {url}: {e}")
        return -1

def get_ip_address(domain):
    """Resolve a domain to its IP address, handling invalid domains gracefully."""
    if not domain or len(domain) > 255:  # Domain names can't be longer than 255 characters
        return None

    try:
        # Encode the domain to IDNA format
        encoded_domain = idna.encode(domain).decode('ascii')
        return socket.gethostbyname(encoded_domain)
    except (socket.gaierror, UnicodeError, idna.IDNAError) as e:
        logging.warning(f"Failed to resolve domain {domain}: {e}")
        return None

SPAMHAUS_DROP_JSON_URL = "https://www.spamhaus.org/drop/drop_v4.json"

def fetch_spamhaus_blacklists():
    """Fetch Spamhaus DROP list (IPv4) and return a set of CIDR blocks."""
    logging.info("Fetching Spamhaus DROP list...")

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(SPAMHAUS_DROP_JSON_URL, headers=headers, timeout=10)
        response.raise_for_status()

        # Check if the response is JSON or NDJSON
        try:
            data = response.json()  # Try parsing as full JSON
            logging.info(f"Spamhaus data parsed as JSON successfully.")
        except json.JSONDecodeError:
            data = response.text.splitlines()  # If fails, treat as NDJSON
            logging.warning(f"Failed to parse full JSON, processing as NDJSON.")

        cidr_blocks = set()
        for entry in data:
            try:
                entry_json = json.loads(entry) if isinstance(entry, str) else entry
                if "cidr" in entry_json:
                    cidr_blocks.add(entry_json["cidr"])
                else:
                    logging.warning(f"Skipping entry without 'cidr': {entry_json}")
            except json.JSONDecodeError as e:
                logging.warning(f"Skipping invalid JSON line: {entry} - {e}")

        logging.info(f"Retrieved {len(cidr_blocks)} CIDR blocks from Spamhaus.")
        logging.info(f"First 5 CIDR blocks: {list(cidr_blocks)[:5]}")  # Print first few entries

        return cidr_blocks

    except Exception as e:
        logging.error(f"Error fetching Spamhaus DROP list: {e}")
        return set()

# Load Spamhaus blacklists into memory
spamhaus_blacklist = fetch_spamhaus_blacklists()

def is_ip_blacklisted(ip):
    """Check if an IP is blacklisted using Spamhaus DROP list."""
    try:
        ip_obj = ipaddress.ip_address(ip)
        for cidr in spamhaus_blacklist:
            if ip_obj in ipaddress.ip_network(cidr, strict=False):
                return 1  # Blacklisted
        return 0  # Not blacklisted
    except ValueError:
        return -1  # Invalid IP

def get_request_url_feature(url):
    """Estimate the request_url feature without accessing the URL."""
    try:
        domain = tldextract.extract(url).registered_domain
        query_params = urlparse(url).query

        # Extract external domains from the query string
        external_domains = set()
        pattern = re.compile(r"(https?://[^\s&]+)")
        
        for match in pattern.findall(query_params):
            resource_domain = tldextract.extract(match).registered_domain
            if resource_domain and resource_domain != domain:
                external_domains.add(resource_domain)

        # If there are any external domains, return 1 (phishing risk)
        return 1 if external_domains else 0

    except Exception:
        return -1  # Assume legitimate if an error occurs


def get_sfh_features_bulk(urls):
    sfh_cache = TTLCache(maxsize=10000, ttl=86400)

    def is_valid_url(url):
        """Validate URL structure before processing"""
        try:
            result = urlparse(url)
            if not all([result.scheme, result.netloc]):
                return False
            # Additional validation for international domains
            if len(result.netloc) > 253:  # Max domain length
                return False
            return True
        except:
            return False

    def fetch_sfh(url):
        if url in sfh_cache:
            return url, sfh_cache[url]

        # Skip invalid URLs immediately
        if not is_valid_url(url):
            return url, -1

        try:
            # Decode percent-encoded URLs first
            decoded_url = urllib.parse.unquote(url)
            response = requests.get(decoded_url, 
                                 timeout=3,
                                 headers={'User-Agent': 'Mozilla/5.0'},
                                 allow_redirects=False)  # Disable redirects
            response.raise_for_status()
            
            domain = tldextract.extract(decoded_url).registered_domain
            soup = BeautifulSoup(response.text, "html.parser")
            
            for form in soup.find_all("form"):
                action = form.get("action", "")
                action_domain = tldextract.extract(urlparse(action).netloc).registered_domain
                
                if not action:
                    sfh_cache[url] = 0
                    return url, 0
                elif action_domain and action_domain != domain:
                    sfh_cache[url] = 2
                    return url, 2

            sfh_cache[url] = 1
            return url, 1

        except (requests.RequestException, UnicodeError, ValueError) as e:
            logging.debug(f"SFH check failed for {url}: {str(e)[:100]}")
            return url, -1

    # Filter URLs before processing
    valid_urls = [url for url in urls if is_valid_url(url)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as exe:
        results = dict(exe.map(fetch_sfh, valid_urls))
    
    # Include invalid URLs with default value
    return {url: results.get(url, -1) for url in urls}

def extract_url_features(urls, tranco_domains):
    parsed_urls = []
    tld_info_list = []

    for url in urls:
        if not url or url.strip() in ('.', '..'):  # Handle empty and dot URLs
            parsed_urls.append(None)
            tld_info_list.append(None)
            continue
        
        try:
            # Add http:// if missing
            if not url.startswith(("http://", "https://")):
                url = "http://" + url
            
            # Basic URL validation
            if not re.match(r'^https?://[^\s/$.?#].[^\s]*$', url, re.IGNORECASE):
                parsed_urls.append(None)
                tld_info_list.append(None)
                continue
                
            parsed_urls.append(urlparse(url))
            tld_info_list.append(tldextract.extract(url))
        except Exception:
            parsed_urls.append(None)
            tld_info_list.append(None)
    
    # Get domain ages only for valid domains
    unique_domains = {f"{tld.domain}.{tld.suffix}" for tld in tld_info_list if tld is not None}
    domain_ages = get_domain_ages_bulk(unique_domains) if unique_domains else {}
    
    # Get SFH features only for valid URLs
    valid_urls_for_sfh = [url for url in urls if url and url.strip() not in ('.', '..')]
    sfh_features = get_sfh_features_bulk(valid_urls_for_sfh) if valid_urls_for_sfh else {}

    features_list = []
    special_chars = "#@-.$*[](){}+;~:'/%?,=&!_"

    for url, parsed_url, tld_info in zip(urls, parsed_urls, tld_info_list):
        # Create default feature set for invalid URLs
        if not url or url.strip() in ('.', '..') or parsed_url is None or tld_info is None:
            features = {
                "url_length": len(url) if url else -1,
                "domain_length": -1,
                "subdomain_length": -1,
                "path_length": -1,
                "query_length": -1,
                "has_https": -1,
                "has_ip": -1,
                "prefix_suffix": -1,
                "domain_age": -1,
                "sub_domain": -1,
                "web_traffic": -1,
                "SFH": -1,
                "url_anchor": -1,
                "request_url": -1,
                "ip_blacklisted": -1,
                "double_forward_slash_redirect": -1,
                "exe": -1,
                "sensitive_words": -1,
                "port_number": -1,
                "free_hosting": -1,
                "count_NAN": -1,
                "number_of_letters": -1,
                "number_of_digits": -1,
                "percentage_special_chars": -1,
                "number_of_special_characters": -1,
            }
            features_list.append(json.dumps(features))
            continue

        effective_domain = f"{tld_info.domain}.{tld_info.suffix}"
        subdomain = tld_info.subdomain

        # Extract IP address from netloc (handling port numbers)
        netloc = parsed_url.netloc
        if ":" in netloc:
            netloc = netloc.split(":")[0]  # Remove port number

        # Check if the netloc is a valid IP address
        has_ip = 1 if is_valid_ip(netloc) else 0

        # Resolve IP address for blacklist check
        resolved_ip = get_ip_address(netloc)
        ip_blacklisted = is_ip_blacklisted(resolved_ip) if resolved_ip else -1

        try:
            port = parsed_url.port if parsed_url.port else -1
        except ValueError:
            port = -1

        # Check if the domain is invalid, but allow IP addresses
        if not effective_domain or len(effective_domain) > 255 or (
            not re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', effective_domain) and not is_valid_ip(netloc)):
            features = {
                "url_length": len(url),
                "domain_length": -1,
                "subdomain_length": -1,
                "path_length": -1,
                "query_length": len(parsed_url.query),
                "has_https": int(parsed_url.scheme == "https"),
                "has_ip": has_ip,
                "prefix_suffix": -1,
                "domain_age": -1,
                "sub_domain": -1,
                "web_traffic": -1,
                "SFH": -1,
                "url_anchor": int(bool(parsed_url.fragment)),
                "request_url": -1,
                "ip_blacklisted": ip_blacklisted,
                "double_forward_slash_redirect": int("//" in parsed_url.path),
                "exe": int(".exe" in parsed_url.path),
                "sensitive_words": int(any(word in url.lower() for word in ["confirm", "account", "bank", "secure", "login", "sign in", "webscr", "submit", "update", "logon", "wp", "cmd", "admin"])),
                "port_number": port, 
                "free_hosting": -1,
                "count_NAN": sum(1 for c in url if not c.isalnum()),
                "number_of_letters": sum(c.isalpha() for c in url),
                "number_of_digits": sum(c.isdigit() for c in url),
                "percentage_special_chars": (sum(url.count(char) for char in special_chars) / len(url)) * 100 if len(url) > 0 else 0,
                "number_of_special_characters": sum(url.count(char) for char in special_chars),
            }
            features_list.append(json.dumps(features))
            continue

        # Path length calculation for valid URLs
        path = parsed_url.path
        if not path:
            path_length = 0
        elif path == "/":
            path_length = 1
        else:
            path_length = len(path)

        features = {
            "url_length": len(url),
            "domain_length": len(effective_domain),
            "subdomain_length": len(subdomain),
            "path_length": path_length,
            "query_length": len(parsed_url.query),
            "has_https": int(parsed_url.scheme == "https"),
            "has_ip": has_ip,
            "prefix_suffix": int("-" in effective_domain),
            "domain_age": domain_ages.get(effective_domain, -1),
            "sub_domain": int(len(subdomain) > 0),
            "web_traffic": get_web_traffic(url, tranco_domains),
            "SFH": sfh_features.get(url, -1),
            "url_anchor": int(bool(parsed_url.fragment)),
            "request_url": get_request_url_feature(url),
            "ip_blacklisted": ip_blacklisted,
            "double_forward_slash_redirect": int("//" in parsed_url.path),
            "exe": int(".exe" in parsed_url.path),
            "sensitive_words": int(any(word in url.lower() for word in ["confirm", "account", "bank", "secure", "login", "sign in", "webscr", "submit", "update", "logon", "wp", "cmd", "admin"])),
            "port_number": port,
            "free_hosting": int(any(host in parsed_url.netloc.lower() for host in [
                "wix.com", "awardspace.com", "wordpress.com", "kamatera.com", "googiehost.com", 
                "x10hosting.com", "github.io", "godaddy.com", "hostinger.com", "freehostia.com", 
                "000webhost.com", "byethost.com", "cloud.google.com", "aws.amazon.com", 
                "hostgator.com", "neocities.org", "infinityfree.net", "weebly.com", 
                "bluehost.com", "siteground.com", "netlify.app", "hostawesome.com", 
                "freehostingnoads.net", "epizy.com", "rf.gd", "vercel.app", "pages.dev", 
                "surge.sh", "onrender.com", "freehosting.com", "freenom.com", "duckdns.org"
            ])),
            "count_NAN": sum(1 for c in url if not c.isalnum()),
            "number_of_letters": sum(c.isalpha() for c in url),
            "number_of_digits": sum(c.isdigit() for c in url),
            "percentage_special_chars": (sum(url.count(char) for char in special_chars) / len(url)) * 100 if len(url) > 0 else 0,
            "number_of_special_characters": sum(url.count(char) for char in special_chars),
        }
        features_list.append(json.dumps(features))

    return pd.Series(features_list)

tranco_domains = preload_tranco_list()
top_100k_tranco_domains = {k: v for k, v in list(tranco_domains.items())[:1048576]}  
broadcast_tranco_domains = spark.sparkContext.broadcast(top_100k_tranco_domains)

@pandas_udf(StringType(), PandasUDFType.SCALAR)
def extract_features_udf(url_series: pd.Series) -> pd.Series:
    tranco = broadcast_tranco_domains.value
    return extract_url_features(url_series.tolist(), tranco)  # Process URLs in bulk

# Load data
file_path = "malicious_urls.csv"
df_spark = spark.read.csv(file_path, header=True, inferSchema=True)

# Standardise column names
if "URLs" in df_spark.columns:
    df_spark = df_spark.withColumnRenamed("URLs", "url")
if "label" in df_spark.columns: 
    df_spark = df_spark.withColumnRenamed("label", "type")
if "Class" in df_spark.columns:
    df_spark = df_spark.withColumnRenamed("Class", "type")
    # Convert labels "good"=0, "bad"=1
    df_spark = df_spark.withColumn("type", 
        when(col("type") == "bad", 1).otherwise(0))
    
# Old dataset compatibility: Ensure "no" column exists (if needed)
if "no" not in df_spark.columns:
    df_spark = df_spark.withColumn("no", monotonically_increasing_id())

# Filter out rows where the URL is None or empty
df_spark = df_spark.filter(col("url").isNotNull() & (col("url") != ""))
# df_spark = df_spark.filter(col("url").isNotNull() & (col("url") != ""))

# Save empty/None URLs to a separate file
empty_urls_df = df_spark.filter(col("url").isNull() | (col("url") == ""))
empty_urls_df.write.mode("overwrite").csv("empty_urls.csv")
# df_spark = spark.read.csv(file_path, header=True, inferSchema=True).repartition(50)

df_spark.show(5)  # Ensure URLs are loaded
df_spark.printSchema()  # Verify URL column exists

# Start time for feature extraction
start_time = time.time()

# Extract features
df_spark_with_features = df_spark.withColumn("features", extract_features_udf(col("url")))

# Define schema for feature extraction
schema = StructType([
    StructField("url_length", IntegerType(), True),
    StructField("domain_length", IntegerType(), True),
    StructField("subdomain_length", IntegerType(), True),
    StructField("path_length", IntegerType(), True),
    StructField("query_length", IntegerType(), True),
    StructField("has_https", IntegerType(), True),
    StructField("has_ip", IntegerType(), True),
    StructField("prefix_suffix", IntegerType(), True),
    StructField("domain_age", IntegerType(), True),
    StructField("sub_domain", IntegerType(), True),
    StructField("web_traffic", DoubleType(), True),
    StructField("SFH", IntegerType(), True),
    StructField("url_anchor", IntegerType(), True),
    StructField("request_url", IntegerType(), True),
    StructField("ip_blacklisted", IntegerType(), True),
    StructField("double_forward_slash_redirect", IntegerType(), True),
    StructField("exe", IntegerType(), True),
    StructField("sensitive_words", IntegerType(), True),
    StructField("port_number", IntegerType(), True),
    StructField("free_hosting", IntegerType(), True),
    StructField("count_NAN", IntegerType(), True),
    StructField("number_of_letters", IntegerType(), True),
    StructField("number_of_digits", IntegerType(), True),
    StructField("percentage_special_chars", DoubleType(), True),  # NEW COMBINED FEATURE
    StructField("number_of_special_characters", IntegerType(), True),  # NEW FEATURE
]) # + [StructField(f"percentage_{char}", DoubleType(), True) for char in "#@-.$*[](){}+;~:'/%?,=&!_"]

# Convert JSON column to structured DataFrame columns
df_spark_with_features = df_spark_with_features.withColumn("features", from_json(col("features"), schema))

# Extract all fields at once
selected_columns = ["no", "url", "type"] + [col("features." + field).alias(field) for field in schema.fieldNames()]
df_spark_with_features = df_spark_with_features.select(*selected_columns).cache()  # Cache intermediate results

# Drop the original JSON column
df_spark_with_features = df_spark_with_features.drop("features")

# Save as a single CSV file
output_dir = "extracted_features"
df_spark_with_features.write.mode("overwrite").parquet("extracted_features.parquet")

# Merge CSV output into a single file
output_file = "extracted_features.csv"
part_files = list(Path(output_dir).glob("part-*.csv"))

if part_files:
    part_file = part_files[0]  # Pick the first file
    shutil.move(str(part_file), output_file)  # Move the file instead of renaming
    shutil.rmtree(output_dir)  # Remove the directory after moving the file

# Stop timing
elapsed_time = time.time() - start_time
print(f"Feature extraction completed and saved to '{output_file}'.")
print(f"Feature extraction took {elapsed_time:.2f} seconds ({elapsed_time // 60:.0f} minutes).")

# Load the Parquet file
df_parquet = spark.read.parquet("extracted_features.parquet")

# Convert to CSV
df_parquet.coalesce(1).write.option("header", "true").csv("extracted_features_csv", mode="overwrite")

output_csv_dir = "extracted_features_csv"
output_csv_file = "extracted_features.csv"
part_files = list(Path(output_csv_dir).glob("part-*.csv"))

if part_files:
    part_file = part_files[0]  # Pick the first part file
    shutil.move(str(part_file), output_csv_file)  # Move and rename it
    shutil.rmtree(output_csv_dir)  # Remove the directory after moving the file

print("Parquet file successfully converted to CSV: extracted_features.csv")

# Stop Spark session
spark.stop()