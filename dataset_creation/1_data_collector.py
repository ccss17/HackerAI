#!/usr/bin/env python3

import os
import json
import logging
import argparse
import shutil
import time
from datetime import datetime, timedelta
from typing import Dict, List, Union, Optional
from pathlib import Path

import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import feedparser
import pandas as pd
import yaml
from github import Github, Auth
from dotenv import load_dotenv
import requests
from malwarebazaar import Bazaar

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CyberDataCollector:
    def __init__(self, output_dir: str = "raw_data"):
        """Initialize the data collector with output directory configuration.

        Required API Keys (set as environment variables):
        - VIRUSTOTAL_API_KEY: Required for VirusTotal API access
        - ALIENVAULT_API_KEY: Required for AlienVault OTX API
        - HTB_API_KEY: Required for HackTheBox API

        Rate Limits:
        - CTFtime API: 30 requests per minute
        - NVD API: 5 requests per 30 seconds
        - VirusTotal API: Depends on subscription tier
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize API clients
        github_token = os.getenv("GITHUB_TOKEN")
        if github_token:
            auth = Auth.Token(github_token)
            self.github_client = Github(auth=auth)
        else:
            self.github_client = None
        self.opencve_auth = (
            os.getenv("OPENCVE_EMAIL"),
            os.getenv("OPENCVE_PASSWORD"),
        )
        self.nvd_api_key = os.getenv("NVD_API_KEY")

        # Load API keys from environment variables
        self.api_keys = {
            "virustotal": os.getenv("VIRUSTOTAL_API_KEY"),
            "alienvault": os.getenv("ALIENVAULT_API_KEY"),
            "hackthebox": os.getenv("HTB_API_KEY"),
            "malpedia": os.getenv("MALPEDIA_API_KEY"),
            "malwarebazaar": os.getenv("MALWAREBAZAAR_API_KEY"),
            "malshare": os.getenv("MALSHARE_API_KEY"),
            "shodan": os.getenv("SHODAN_API_KEY"),
            "phishtank": os.getenv("PHISHTANK_API_KEY"),
            "rootme": os.getenv("ROOTME_API_KEY"),
            "threatfox": os.getenv("THREATFOX_API_KEY"),
        }

        # Initialize rate limiting
        self.rate_limits = {
            "nvd_cve": {"requests": 5, "period": 30},
            "ctftime": {"requests": 30, "period": 60},
            "github": {"requests": 60, "period": 3600},  # GitHub API limit
            "virustotal": {"requests": 4, "period": 60},
            "shodan": {"requests": 1, "period": 1},
            "malshare": {"requests": 25, "period": 60},
        }
        self.last_request_time = {}

        # Add request timeout settings
        self.timeouts = {
            "default": 30,
            "download": 180,  # Longer timeout for downloading larger files
            "scraping": 60,  # Longer timeout for web scraping
        }

        # Add retry configurations
        self.retry_config = {
            "max_retries": 3,
            "base_delay": 5,
            "max_delay": 60,
            "exponential_backoff": True,
        }

        # API endpoints and configurations
        self.endpoints = {
            # NIST and CVE Sources
            "nvd_cve": "https://services.nvd.nist.gov/rest/json/cves/2.0",
            "opencve": "https://app.opencve.io/api/cve",
            "nist_standards": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.json",
            "mitre_attack": "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json",
            "mitre_capec": "https://capec.mitre.org/data/xml/views/3000.xml",
            # Threat Intelligence Feeds
            "alienvault_otx": "https://otx.alienvault.com/api/v1/pulses/subscribed",
            "threatfox_api": "https://threatfox-api.abuse.ch/api/v1/",
            # Security Advisories
            "microsoft_security": "https://api.msrc.microsoft.com/cvrf/v2.0/updates",
            "ubuntu_usn": "https://ubuntu.com/security/notices/rss.xml",
            "redhat_security": "https://access.redhat.com/hydra/rest/securitydata/cve.json",
            # Research and Reports
            "arxiv_cs_crypto": "http://export.arxiv.org/api/query?search_query=cat:cs.CR&max_results=100",
            "exploit_db": "https://www.exploit-db.com/download/",
            # Malware Information
            "malware_bazaar": "https://mb-api.abuse.ch/api/v1/",
            "virustotal": "https://www.virustotal.com/api/v3",
            "malpedia": "https://malpedia.caad.fkie.fraunhofer.de/api/v1/",
            "malshare": "https://malshare.com/api.php",
            "thezoo": "https://github.com/ytisf/theZoo/raw/master/malware.yml",
            "vxug": "https://vx-underground.org/samples.html",
            # CTF Resources
            "ctftime": "https://ctftime.org/api/v1/events/",
            "root_me": "https://api.www.root-me.org/challenges",
            "hackthebox": "https://labs.hackthebox.com/api/v4/challenge/list",
            # Security Testing Resources
            "metasploit_modules": "https://raw.githubusercontent.com/rapid7/metasploit-framework/master/modules/",
            "pentesterlab": "https://pentesterlab.com/exercises/api/v1/",
            "vulnhub": "https://www.vulnhub.com/api/v1/entries/",
            "offensive_security": "https://offsec.tools/api/tools",
            "securitytube": "https://www.securitytube.net/api/v1/videos",
            "pentestmonkey": "https://github.com/pentestmonkey/php-reverse-shell/raw/master/php-reverse-shell.php",
            "payloadsallthethings": "https://raw.githubusercontent.com/swisskyrepo/PayloadsAllTheThings/master/",
            # Social Engineering Resources
            "phishtank": "https://phishtank.org/phish_search.php?valid=y&active=all&Search=Search",
            "openphish": "https://openphish.com/feed.txt",
            "social_engineer_toolkit": "https://github.com/trustedsec/social-engineer-toolkit/raw/master/src/templates/",
            "gophish": "https://github.com/gophish/gophish/raw/master/templates/",
            # DoS/DDoS Resources
            "ddosdb": "https://ddosdb.org/api/v1/",
            "netscout_atlas": "https://atlas.netscout.com/api/v2/",
            # MITM & Injection Resources
            "bettercap": "https://raw.githubusercontent.com/bettercap/bettercap/master/modules/",
            "sqlmap": "https://raw.githubusercontent.com/sqlmapproject/sqlmap/master/data/",
            "nosqlmap": "https://raw.githubusercontent.com/codingo/NoSQLMap/master/attacks/",
            # Zero-Day & Password Resources
            "zerodayinitiative": "https://www.zerodayinitiative.com/rss/published/",
            "project_zero": "https://bugs.chromium.org/p/project-zero/issues/list?rss=true",
            "rockyou": "https://github.com/danielmiessler/SecLists/raw/master/Passwords/Leaked-Databases/",
            "hashcat": "https://hashcat.net/hashcat/",
            # IoT Security Resources
            "iot_vulndb": "https://www.exploit-db.com/download/iot/",
            "iot_sentinel": "https://iotsentinel.csec.ch/api/v1/",
            "shodan_iot": "https://api.shodan.io/shodan/host/search?key={}&query=iot",
        }

        # Initialize session for better performance
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "CyberLLMInstruct-DataCollector/1.0"}
        )

    def _check_rate_limit(self, endpoint: str) -> None:
        """
        Implement rate limiting for APIs.
        Sleeps if necessary to respect rate limits.
        """
        if endpoint not in self.rate_limits:
            return

        current_time = time.time()
        if endpoint in self.last_request_time:
            elapsed = current_time - self.last_request_time[endpoint]
            limit = self.rate_limits[endpoint]
            if elapsed < (limit["period"] / limit["requests"]):
                sleep_time = (limit["period"] / limit["requests"]) - elapsed
                logger.debug(
                    f"Rate limiting {endpoint}, sleeping for {sleep_time:.2f}s"
                )
                time.sleep(sleep_time)

        self.last_request_time[endpoint] = current_time

    def _make_request(
        self,
        endpoint: str,
        url: str,
        params: Dict = None,
        headers: Dict = None,
        timeout: int = None,
        method: str = "get",
        data: Dict = None,
        auth=None,
    ) -> Optional[requests.Response]:
        """
        Enhanced request method with better error handling and retries.
        """
        self._check_rate_limit(endpoint)

        if headers is None:
            headers = {}

        # Set auth for OpenCVE
        if endpoint == "opencve" and not auth:
            auth = self.opencve_auth

        # Add API keys to headers based on endpoint
        for key, value in self.api_keys.items():
            if endpoint.startswith(key) and value:
                if key == "virustotal":
                    headers["x-apikey"] = value
                elif key == "alienvault":
                    headers["X-OTX-API-KEY"] = value
                # ... add other API key headers as needed

        timeout = timeout or self.timeouts["default"]
        retry_count = 0
        last_error = None

        while retry_count < self.retry_config["max_retries"]:
            try:
                if method.lower() == "get":
                    response = self.session.get(
                        url,
                        params=params,
                        headers=headers,
                        timeout=timeout,
                        auth=auth,
                    )
                elif method.lower() == "post":
                    # Check if we need to send form data or JSON data
                    content_type = headers.get("Content-Type", "")
                    if "application/x-www-form-urlencoded" in content_type:
                        response = self.session.post(
                            url,
                            params=params,
                            headers=headers,
                            data=data,
                            timeout=timeout,
                            auth=auth,
                        )
                    else:
                        response = self.session.post(
                            url,
                            params=params,
                            headers=headers,
                            json=data,
                            timeout=timeout,
                            auth=auth,
                        )

                response.raise_for_status()
                return response

            except requests.exceptions.RequestException as e:
                last_error = e
                retry_count += 1

                if retry_count == self.retry_config["max_retries"]:
                    break

                # Calculate delay with exponential backoff
                if self.retry_config["exponential_backoff"]:
                    delay = min(
                        self.retry_config["base_delay"]
                        * (2 ** (retry_count - 1)),
                        self.retry_config["max_delay"],
                    )
                else:
                    delay = self.retry_config["base_delay"]

                logger.warning(
                    f"Request failed (attempt {retry_count}/{self.retry_config['max_retries']}): {str(e)}"
                )
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)

        logger.error(f"All retry attempts failed for {url}: {str(last_error)}")
        return None

    def fetch_cve_data(
        self, start_index: int = 0, results_per_page: int = 2000
    ) -> Optional[Dict]:
        """
        Fetch CVE data from NVD database.

        Note: Implements rate limiting of 5 requests per 30 seconds
        """
        try:
            params = {
                "startIndex": start_index,
                "resultsPerPage": results_per_page,
            }
            response = self._make_request(
                "nvd_cve", self.endpoints["nvd_cve"], params=params
            )
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching CVE data: {str(e)}")
            return None

    def fetch_opencve_data(self, limit: int = 100) -> Optional[Dict]:
        """
        Fetch CVE data from the OpenCVE API.

        Args:
            limit: Maximum number of CVEs to fetch

        Returns:
            Dictionary containing CVE data or None if failed

        Note: Uses basic authentication with the provided OpenCVE credentials.
        """
        try:
            all_cves = []
            page = 1

            # Fetch pages until we reach the limit or there are no more pages
            while len(all_cves) < limit:
                params = {"page": page}
                response = self._make_request(
                    "opencve", self.endpoints["opencve"], params=params
                )

                if not response:
                    break

                data = response.json()
                results = data.get("results", [])

                if not results:
                    break

                all_cves.extend(results)

                # Check if there's a next page
                if not data.get("next"):
                    break

                page += 1

                # Limit the number of CVEs
                if len(all_cves) >= limit:
                    all_cves = all_cves[:limit]
                    break

            # Get detailed information for each CVE
            detailed_cves = []
            for cve in all_cves[
                :10
            ]:  # Limit detailed lookups to avoid rate limiting
                cve_id = cve.get("cve_id")
                if cve_id:
                    detailed_url = f"{self.endpoints['opencve']}/{cve_id}"
                    detailed_response = self._make_request(
                        "opencve", detailed_url
                    )

                    if detailed_response:
                        detailed_cves.append(detailed_response.json())

            return {
                "summary": all_cves,
                "detailed": detailed_cves,
                "count": len(all_cves),
                "timestamp": datetime.now().isoformat(),
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching OpenCVE data: {str(e)}")
            return None

    def fetch_nist_standards(self) -> Optional[Dict]:
        """
        Fetch NIST cyber security standards.

        Returns:
            Dictionary containing NIST standards or None if failed
        """
        logger.warning("NIST standards fetching is disabled due to URL issues")
        return None

    def fetch_mitre_attack(self) -> Optional[Dict]:
        """Fetch MITRE ATT&CK framework data."""
        try:
            response = self.session.get(self.endpoints["mitre_attack"])
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching MITRE ATT&CK data: {str(e)}")
            return None

    def fetch_capec_data(self) -> Optional[Dict]:
        """Fetch MITRE CAPEC (Common Attack Pattern Enumeration and Classification) data."""
        try:
            response = self.session.get(self.endpoints["mitre_capec"])
            response.raise_for_status()
            return {"xml_data": response.text}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching CAPEC data: {str(e)}")
            return None

    def fetch_ubuntu_security_notices(self) -> Optional[Dict]:
        """Fetch Ubuntu Security Notices."""
        try:
            feed = feedparser.parse(self.endpoints["ubuntu_usn"])
            return {"entries": feed.entries}
        except Exception as e:
            logger.error(f"Error fetching Ubuntu Security Notices: {str(e)}")
            return None

    def fetch_arxiv_papers(self) -> Optional[Dict]:
        """Fetch recent cyber security papers from arXiv."""
        try:
            response = self.session.get(self.endpoints["arxiv_cs_crypto"])
            response.raise_for_status()
            feed = feedparser.parse(response.text)
            return {"papers": feed.entries}
        except Exception as e:
            logger.error(f"Error fetching arXiv papers: {str(e)}")
            return None

    def fetch_redhat_security(self) -> Optional[Dict]:
        """Fetch Red Hat Security Data from Hydra REST API."""
        try:
            # Use the official Hydra REST API endpoint
            response = self.session.get(self.endpoints["redhat_security"])
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Red Hat Security data: {str(e)}")
            return None

    def fetch_microsoft_security(self) -> Optional[Dict]:
        """Fetch Microsoft Security Updates."""
        try:
            response = self.session.get(self.endpoints["microsoft_security"])
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(
                f"Error fetching Microsoft Security Updates: {str(e)}"
            )
            return None

    def fetch_alienvault_otx(self) -> Optional[Dict]:
        """Fetch threat intelligence from AlienVault OTX."""
        try:
            if not self.api_keys.get("alienvault"):
                logger.warning("AlienVault API key not set, skipping")
                return None

            headers = {"X-OTX-API-KEY": self.api_keys["alienvault"]}
            response = self.session.get(
                self.endpoints["alienvault_otx"], headers=headers
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching AlienVault OTX data: {str(e)}")
            return None

    def fetch_threatfox(self) -> Optional[Dict]:
        """Fetch threat indicators from ThreatFox (abuse.ch).

        Note: Requires API key from https://threatfox.abuse.ch/
        """
        try:
            if not self.api_keys.get("threatfox"):
                logger.warning("ThreatFox API key not set, skipping")
                return None

            # ThreatFox API requires JSON POST with API key
            payload = {"query": "get_iocs", "days": 7}
            headers = {
                "API-KEY": self.api_keys["threatfox"],
                "Content-Type": "application/json",
            }
            response = self.session.post(
                self.endpoints["threatfox_api"], json=payload, headers=headers
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching ThreatFox data: {str(e)}")
            return None

    def fetch_malware_bazaar(self) -> Optional[Dict]:
        """Fetch recent malware samples from MalwareBazaar.

        Note: Requires API key for access.
        Requires: pip install malwarebazaar
        """
        try:
            if not self.api_keys.get("malwarebazaar"):
                logger.warning("MalwareBazaar API key not set, skipping")
                return None

            # Use the official SDK
            b = Bazaar(api_key=self.api_keys["malwarebazaar"])
            response = b.query_recent()
            return response
        except Exception as e:
            logger.error(f"Error fetching MalwareBazaar data: {str(e)}")
            return None

    def fetch_virustotal_info(self, file_hashes: Optional[List[str]] = None) -> Optional[Dict]:
        """Fetch file reports from VirusTotal v3.

        Behavior:
        - If VIRUSTOTAL_HASHES env var is set (comma-separated), use it.
        - If file_hashes is provided, use it.
        - If neither is provided, return a helpful status object.
        """
        try:
            if not self.api_keys.get("virustotal"):
                logger.warning("VirusTotal API key not set, skipping")
                return None

            # Resolve hashes from arg or environment
            if not file_hashes:
                hashes_env = os.getenv("VIRUSTOTAL_HASHES", "")
                file_hashes = [h.strip() for h in hashes_env.split(",") if h.strip()]

            if not file_hashes:
                logger.info(
                    "VirusTotal configured but no hashes provided. "
                    "Set VIRUSTOTAL_HASHES=hash1,hash2 to fetch reports."
                )
                return {
                    "status": "configured",
                    "note": "Set VIRUSTOTAL_HASHES=sha256_1,sha256_2 to fetch file reports.",
                }

            reports = []
            for h in file_hashes:
                url = f"{self.endpoints['virustotal']}/files/{h}"
                resp = self._make_request("virustotal", url, timeout=self.timeouts["default"])
                if not resp:
                    logger.warning(f"VirusTotal: no response for hash {h}")
                    continue
                try:
                    data = resp.json()
                except Exception:
                    data = {"raw": resp.text}
                reports.append({"hash": h, "report": data})

            return {
                "reports": reports,
                "count": len(reports),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error with VirusTotal: {str(e)}")
            return None

    def fetch_malshare_samples(self) -> Optional[Dict]:
        """Fetch malware samples list from MalShare."""
        try:
            if not self.api_keys.get("malshare"):
                logger.warning("MalShare API key not set, skipping")
                return None

            params = {
                "api_key": self.api_keys["malshare"],
                "action": "getlist",
            }
            response = self.session.get(
                self.endpoints["malshare"], params=params
            )
            response.raise_for_status()
            return {"samples": response.text.splitlines()}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching MalShare data: {str(e)}")
            return None

    def fetch_root_me_challenges(self) -> Optional[Dict]:
        """Fetch challenges from Root-Me platform (requires API key)."""
        try:
            if not self.api_keys.get("rootme"):
                logger.warning("Root-Me API key not set, skipping")
                return None

            headers = {"Authorization": f"Bearer {self.api_keys['rootme']}"}
            response = self.session.get(
                self.endpoints["root_me"], headers=headers
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Root-Me challenges: {str(e)}")
            return None

    def fetch_hackthebox_challenges(self) -> Optional[Dict]:
        """Fetch challenges from HackTheBox (requires API key)."""
        try:
            if not self.api_keys.get("hackthebox"):
                logger.warning("HackTheBox API key not set, skipping")
                return None

            headers = {
                "Authorization": f"Bearer {self.api_keys['hackthebox']}"
            }
            response = self.session.get(
                self.endpoints["hackthebox"], headers=headers
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching HackTheBox challenges: {str(e)}")
            return None

    def fetch_shodan_data(self) -> Optional[Dict]:
        """Fetch IoT/network data from Shodan (requires API key)."""
        try:
            if not self.api_keys.get("shodan"):
                logger.warning("Shodan API key not set, skipping")
                return None

            # Example: Search for common IoT devices
            url = f"https://api.shodan.io/shodan/host/search?key={self.api_keys['shodan']}&query=port:22"
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Shodan data: {str(e)}")
            return None

    def fetch_openphish_feed(self) -> Optional[Dict]:
        """Fetch phishing URLs from OpenPhish feed."""
        try:
            response = self.session.get(self.endpoints["openphish"])
            response.raise_for_status()
            urls = response.text.splitlines()
            return {"phishing_urls": urls, "count": len(urls)}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching OpenPhish feed: {str(e)}")
            return None

    def fetch_exploit_db_recent(self) -> Optional[Dict]:
        """Fetch recent exploits metadata from Exploit-DB."""
        try:
            # Exploit-DB provides CSV files
            csv_url = "https://gitlab.com/exploit-database/exploitdb/-/raw/main/files_exploits.csv"
            response = self.session.get(csv_url)
            response.raise_for_status()

            # Parse CSV and return recent exploits
            from io import StringIO
            import csv

            csv_data = csv.DictReader(StringIO(response.text))
            exploits = list(csv_data)[-100:]  # Last 100 exploits
            return {"exploits": exploits}
        except Exception as e:
            logger.error(f"Error fetching Exploit-DB data: {str(e)}")
            return None

    def fetch_project_zero_issues(self, limit: int = 200) -> Optional[Dict]:
        """Fetch recent issues from Google Project Zero (RSS)."""
        try:
            # Request with our session/UA and ask for more items to avoid empty feeds
            url = f"{self.endpoints['project_zero']}&num={limit}&can=1"
            resp = self.session.get(url, timeout=self.timeouts["scraping"])
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)
            entries = getattr(feed, "entries", [])

            # Normalize a few useful fields from the feed
            issues = [
                {
                    "title": e.get("title"),
                    "link": e.get("link"),
                    "published": e.get("published"),
                    "summary": (
                        BeautifulSoup(e.get("summary", ""), "html.parser").get_text()
                        if e.get("summary")
                        else None
                    ),
                }
                for e in entries
            ]

            return {
                "issues": issues,
                "count": len(issues),
                "source": url,
            }
        except Exception as e:
            logger.error(f"Error fetching Project Zero issues: {str(e)}")
            return None

    def fetch_zerodayinitiative(self) -> Optional[Dict]:
        """Fetch advisories from Zero Day Initiative."""
        try:
            response = feedparser.parse(self.endpoints["zerodayinitiative"])
            return {
                "advisories": response.entries
                if hasattr(response, "entries")
                else []
            }
        except Exception as e:
            logger.error(f"Error fetching Zero Day Initiative data: {str(e)}")
            return None

    def fetch_github_security_advisories(self) -> Optional[Dict]:
        """Fetch security advisories from GitHub."""
        try:
            if not os.getenv("GITHUB_TOKEN"):
                logger.warning("GitHub token not set, skipping")
                return None

            # Use GitHub GraphQL API to fetch security advisories
            query = """
            {
              securityVulnerabilities(first: 100, orderBy: {field: UPDATED_AT, direction: DESC}) {
                nodes {
                  advisory {
                    summary
                    description
                    severity
                    publishedAt
                  }
                  package {
                    name
                  }
                }
              }
            }
            """

            headers = {"Authorization": f"token {os.getenv('GITHUB_TOKEN')}"}
            response = self.session.post(
                "https://api.github.com/graphql",
                json={"query": query},
                headers=headers,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(
                f"Error fetching GitHub security advisories: {str(e)}"
            )
            return None

    def fetch_malware_data(self) -> Optional[Dict]:
        """
        Fetch malware data from MalwareBazaar.
        """
        return self.fetch_malware_bazaar()

    def fetch_social_engineering_data(self) -> Optional[Dict]:
        """
        Fetch social engineering data from OpenPhish.
        """
        return self.fetch_openphish_feed()

    def scrape_security_articles(self, url: str) -> Optional[Dict]:
        """
        Scrape cyber security articles from provided URL.

        Args:
            url: URL to scrape

        Returns:
            Dictionary containing scraped data or None if failed
        """
        try:
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract relevant information (customize based on website structure)
            data = {
                "title": soup.title.string if soup.title else None,
                "text": soup.get_text(),
                "url": url,
                "timestamp": datetime.now().isoformat(),
            }
            return data
        except (requests.exceptions.RequestException, AttributeError) as e:
            logger.error(f"Error scraping article from {url}: {str(e)}")
            return None

    def save_data(
        self, data: Union[Dict, List], source: str, format: str = "json"
    ) -> bool:
        """
        Enhanced save_data method with better error handling and backup.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.output_dir / f"{source}_{timestamp}.{format}"

            # Create backup directory
            backup_dir = self.output_dir / "backups"
            backup_dir.mkdir(exist_ok=True)

            # Save data with proper encoding and error handling
            if format == "json":
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())  # Ensure data is written to disk

            elif format == "xml":
                # Improved XML handling
                root = ET.Element("data")
                self._dict_to_xml(data, root)
                tree = ET.ElementTree(root)
                tree.write(filename, encoding="utf-8", xml_declaration=True)

            elif format == "yaml":
                with open(filename, "w", encoding="utf-8") as f:
                    yaml.dump(
                        data, f, allow_unicode=True, default_flow_style=False
                    )
                    f.flush()
                    os.fsync(f.fileno())

            elif format == "csv":
                df = pd.DataFrame(data)
                df.to_csv(filename, index=False, encoding="utf-8")

            # Create backup
            backup_file = backup_dir / f"{source}_{timestamp}_backup.{format}"
            shutil.copy2(filename, backup_file)

            logger.info(
                f"Successfully saved data to {filename} with backup at {backup_file}"
            )
            return True

        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            return False

    def _dict_to_xml(
        self, data: Union[Dict, List, str, int, float], parent: ET.Element
    ):
        """Helper method for converting dictionary to XML."""
        if isinstance(data, dict):
            for key, value in data.items():
                child = ET.SubElement(parent, str(key))
                self._dict_to_xml(value, child)
        elif isinstance(data, (list, tuple)):
            for item in data:
                child = ET.SubElement(parent, "item")
                self._dict_to_xml(item, child)
        else:
            parent.text = str(data)

    def fetch_ctf_data(self) -> Optional[Dict]:
        """
        Fetch CTF event data and challenges from various platforms.

        Returns:
            Dictionary containing CTF data or None if failed
        """
        try:
            # Get upcoming and ongoing CTF events from CTFtime
            # CTFtime API requires start and end time parameters
            start_time = datetime.now()
            end_time = start_time + timedelta(
                days=90
            )  # Get events for next 90 days

            params = {
                "start": int(start_time.timestamp()),
                "finish": int(end_time.timestamp()),
                "limit": 100,
            }

            response = self.session.get(
                self.endpoints["ctftime"], params=params
            )
            response.raise_for_status()
            ctftime_events = response.json()

            # Compile CTF data from different sources
            ctf_data = {
                "ctftime_events": ctftime_events,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "source": "CTFtime API",
                    "event_timeframe": f"{start_time.date()} to {end_time.date()}",
                },
            }

            return ctf_data

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching CTF data: {str(e)}")
            return None

    def fetch_security_testing_resources(self) -> Optional[Dict]:
        """
        Fetch security testing scripts and resources from educational sources.

        Note: Some endpoints may be blocked by corporate firewalls or security policies.
        GitHub rate limits apply for raw.githubusercontent.com requests.
        """
        logger.warning(
            "Security testing resources fetching is disabled due to URL issues"
        )
        return None


def main():
    """Main function to process command-line arguments and run data collection."""
    description = """
    Collect cybersecurity data from various sources.
    
    Public sources (no API key required):
    - cve_data: CVE vulnerability data from NVD
    - mitre_attack: MITRE ATT&CK framework data
    - capec_data: Common Attack Pattern Enumeration and Classification data
    - ubuntu_security: Ubuntu Security Notices
    - arxiv_papers: Recent cybersecurity papers from arXiv
    - redhat_security: Red Hat Security blog feed
    - microsoft_security: Microsoft Security Updates
    - ctf_data: CTF event data from CTFtime
    - threatfox: Threat indicators from ThreatFox
    - malware_bazaar: Malware samples from MalwareBazaar
    - openphish: Phishing URLs from OpenPhish
    - exploit_db: Recent exploits from Exploit-DB
    - project_zero: Issues from Google Project Zero
    - zerodayinitiative: Advisories from Zero Day Initiative
    
    Sources requiring API keys (optional):
    - opencve_data: OpenCVE API (requires OPENCVE_EMAIL, OPENCVE_PASSWORD)
    - alienvault_otx: AlienVault OTX (requires ALIENVAULT_API_KEY)
    - virustotal: VirusTotal (requires VIRUSTOTAL_API_KEY)
    - malshare: MalShare (requires MALSHARE_API_KEY)
    - root_me: Root-Me challenges (public)
    - hackthebox: HackTheBox (requires HTB_API_KEY)
    - shodan: Shodan IoT data (requires SHODAN_API_KEY)
    - github_advisories: GitHub Security Advisories (requires GITHUB_TOKEN)
    
    Disabled sources (known issues):
    - nist_standards: NIST cybersecurity standards (URL issues)
    - security_testing: Security testing resources (URL issues)
    """

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        help="List of sources to fetch data from, space-separated",
    )
    parser.add_argument(
        "--output-dir",
        default="raw_data",
        help="Directory to save collected data",
    )

    args = parser.parse_args()

    collector = CyberDataCollector(output_dir=args.output_dir)

    # Define all available sources (public + API-based)
    all_sources = {
        # Public vulnerability & attack pattern databases
        "cve_data": collector.fetch_cve_data,
        "mitre_attack": collector.fetch_mitre_attack,
        "capec_data": collector.fetch_capec_data,
        # Security advisories & updates
        "ubuntu_security": collector.fetch_ubuntu_security_notices,
        "redhat_security": collector.fetch_redhat_security,
        "microsoft_security": collector.fetch_microsoft_security,
        # Research & publications
        "arxiv_papers": collector.fetch_arxiv_papers,
        # Threat intelligence (public)
        "threatfox": collector.fetch_threatfox,
        "malware_bazaar": collector.fetch_malware_bazaar,
        "openphish": collector.fetch_openphish_feed,
        # Exploit & vulnerability research
        "exploit_db": collector.fetch_exploit_db_recent,
        "project_zero": collector.fetch_project_zero_issues,
        "zerodayinitiative": collector.fetch_zerodayinitiative,
        # CTF & training platforms
        "ctf_data": collector.fetch_ctf_data,
        "root_me": collector.fetch_root_me_challenges,
        # API-key required sources (gracefully skip if key not present)
        "opencve_data": collector.fetch_opencve_data,
        "alienvault_otx": collector.fetch_alienvault_otx,
        "virustotal": collector.fetch_virustotal_info,
        "malshare": collector.fetch_malshare_samples,
        "hackthebox": collector.fetch_hackthebox_challenges,
        "shodan": collector.fetch_shodan_data,
        "github_advisories": collector.fetch_github_security_advisories,
    }

    # Disabled sources (known issues - won't work without fixes)
    disabled_sources = {
        "nist_standards": collector.fetch_nist_standards,
        "security_testing": collector.fetch_security_testing_resources,
    }

    # If specific sources are provided, use only those
    sources_to_fetch = {}
    if args.sources:
        for source in args.sources:
            if source in all_sources:
                sources_to_fetch[source] = all_sources[source]
            elif source in disabled_sources:
                sources_to_fetch[source] = disabled_sources[source]
                logger.warning(f"Including disabled source: {source}")
            elif source == "all":
                sources_to_fetch = all_sources
                break
            else:
                logger.warning(f"Unknown source: {source}, ignoring")
    else:
        # If no sources specified, use all working ones
        sources_to_fetch = all_sources

    logger.info(f"Collecting data from {len(sources_to_fetch)} sources")

    for source_name, fetch_function in sources_to_fetch.items():
        logger.info(f"Fetching data from {source_name}...")
        data = fetch_function()
        if data:
            collector.save_data(data, source_name)
        else:
            logger.warning(f"No data retrieved from {source_name}")


if __name__ == "__main__":
    main()
