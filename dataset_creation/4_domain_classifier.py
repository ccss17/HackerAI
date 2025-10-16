#!/usr/bin/env python3

import json
import logging
import requests
import time
import threading
import signal
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CyberDomainClassifier:
    def __init__(
        self,
        input_dir: str = "structured_data",
        output_dir: str = "domain_classified",
        ollama_model: str = "gemma3",
        ollama_port: int = 11434,
        max_workers: int = 4,
    ):
        """Initialize the domain classifier with directory configurations."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ollama_model = ollama_model
        self.ollama_url = f"http://localhost:{ollama_port}/api/generate"
        self.max_workers = max(1, max_workers)
        self._file_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "domain_counts": {},
        }

        # Define cybersecurity domains
        self.domains = [
            "malware",
            "phishing",
            "zero_day",
            "iot_security",
            "web_security",
            "network_security",
            "vulnerability_management",
            "cloud_security",
            "cryptography",
            "identity_access_management",
        ]

        # Verify Ollama connection
        if not self._verify_ollama_connection():
            logger.error(f"Could not connect to Ollama at {self.ollama_url}")
            raise ConnectionError(
                f"Could not connect to Ollama at {self.ollama_url}"
            )

    def _verify_ollama_connection(self) -> bool:
        """Verify that Ollama is running and accessible."""
        try:
            response = requests.get(
                f"http://localhost:{self.ollama_url.split(':')[-1].split('/')[0]}/api/tags",
                timeout=5,
            )
            return response.status_code == 200
        except Exception:
            return False

    def call_ollama(
        self, prompt: str, system_prompt: str, max_retries: int = 3
    ) -> Optional[str]:
        """Call Ollama API with retry mechanism and exponential backoff."""
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.ollama_url,
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "system": system_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,  # Very low temperature for consistent JSON
                            "num_predict": 200,  # Shorter response to avoid timeouts
                            "top_p": 0.9,
                            "repeat_penalty": 1.1,
                            "stop": [
                                "```",
                                "\n\n",
                            ],  # Stop at markdown or double newlines
                        },
                    },
                    timeout=120,  # Reduced timeout to fail faster
                )
                response.raise_for_status()
                return response.json()["response"]
            except requests.exceptions.Timeout:
                logger.warning(
                    f"Attempt {attempt + 1} timed out after 30 seconds"
                )
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            except Exception as e:
                logger.warning(
                    f"Unexpected error in attempt {attempt + 1}: {str(e)}"
                )

            if attempt < max_retries - 1:
                # Exponential backoff
                time.sleep(2**attempt)
                continue
        return None

    def classify_entry(self, entry: Dict) -> Dict:
        """Classify a single entry using Ollama."""
        # Create prompt for classification
        prompt = f"""Analyze this cybersecurity content and classify it into one or more of these domains: {", ".join(self.domains)}

Content:
Instruction: {entry.get("instruction", "")}
Response: {entry.get("response", "")}

Return ONLY a valid JSON array with domain classifications. Format:
[{{"domain": "domain_name", "confidence": 0.95}}]

Requirements:
- Use only the provided domains: {", ".join(self.domains)}
- Confidence values between 0.0 and 1.0
- Return valid JSON only, no other text
- Include only domains with confidence > 0.2"""

        system_prompt = """You are a cybersecurity domain classification expert. Analyze the content and return ONLY a valid JSON array with domain classifications. Do not include any explanatory text, markdown formatting, or code blocks. Return only the JSON array."""

        response = self.call_ollama(prompt, system_prompt)

        if response:
            try:
                # Clean and extract JSON from response
                cleaned_response = self._clean_ollama_response(response)
                classifications = self._parse_classification_json(
                    cleaned_response
                )

                if classifications:
                    entry["domains"] = classifications
                    entry["primary_domain"] = classifications[0]["domain"]
                else:
                    # Fallback: try to classify based on content keywords
                    fallback_domain = self._classify_by_keywords(entry)
                    entry["domains"] = [
                        {"domain": fallback_domain, "confidence": 0.5}
                    ]
                    entry["primary_domain"] = fallback_domain
            except Exception as e:
                logger.error(f"Error parsing classification JSON: {str(e)}")
                # Fallback classification
                fallback_domain = self._classify_by_keywords(entry)
                entry["domains"] = [
                    {"domain": fallback_domain, "confidence": 0.5}
                ]
                entry["primary_domain"] = fallback_domain
        else:
            # Fallback classification when no response
            fallback_domain = self._classify_by_keywords(entry)
            entry["domains"] = [{"domain": fallback_domain, "confidence": 0.5}]
            entry["primary_domain"] = fallback_domain

        return entry

    def _clean_ollama_response(self, response: str) -> str:
        """Clean and extract JSON from Ollama response with robust error handling."""
        if not response or not isinstance(response, str):
            return "[]"

        # Remove markdown code blocks and clean up
        response = response.replace("```json", "").replace("```", "").strip()

        # Remove control characters that cause JSON parsing issues
        response = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", response)

        # Find JSON array boundaries
        json_start = response.find("[")
        json_end = response.rfind("]") + 1

        if json_start >= 0 and json_end > json_start:
            json_content = response[json_start:json_end]
        else:
            # Try to find any JSON-like structure
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_content = "[" + response[json_start:json_end] + "]"
            else:
                return "[]"

        # Clean up common issues
        json_content = json_content.strip()

        # Remove any remaining control characters
        json_content = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", json_content)

        # Remove trailing commas before closing brackets
        json_content = re.sub(r",(\s*[}\]])", r"\1", json_content)

        # Fix missing quotes around domain names
        json_content = re.sub(
            r'"domain":\s*([^"]+?)(?=\s*[,}])', r'"domain": "\1"', json_content
        )

        # Fix missing quotes around confidence values
        json_content = re.sub(
            r'"confidence":\s*([0-9.]+)', r'"confidence": \1', json_content
        )

        # Fix common JSON syntax issues
        json_content = re.sub(
            r"([,{]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', json_content
        )
        json_content = re.sub(
            r":\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*([,}])", r': "\1"\2', json_content
        )

        return json_content

    def _parse_classification_json(self, json_content: str) -> List[Dict]:
        """Parse classification JSON with robust error handling and fallback."""
        try:
            classifications = json.loads(json_content)

            # Validate and clean the classifications
            valid_classifications = []
            for item in classifications:
                if isinstance(item, dict):
                    domain = item.get("domain", "").strip()
                    confidence = item.get("confidence", 0)

                    # Validate domain is in our list
                    if domain in self.domains and isinstance(
                        confidence, (int, float)
                    ):
                        if (
                            confidence > 0.2
                        ):  # Only include high confidence classifications
                            valid_classifications.append(
                                {
                                    "domain": domain,
                                    "confidence": float(confidence),
                                }
                            )

            # Sort by confidence descending
            valid_classifications.sort(
                key=lambda x: x["confidence"], reverse=True
            )
            return valid_classifications

        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {str(e)}")
            # Try to extract domains using regex as fallback
            return self._extract_domains_fallback(json_content)
        except Exception as e:
            logger.warning(f"Error parsing classifications: {str(e)}")
            return self._extract_domains_fallback(json_content)

    def _extract_domains_fallback(self, text: str) -> List[Dict]:
        """Fallback method to extract domains when JSON parsing fails."""
        try:
            # Look for domain patterns in the text
            domain_matches = []
            for domain in self.domains:
                # Look for domain mentions with confidence scores
                pattern = (
                    rf'["\']?{re.escape(domain)}["\']?\s*[:,]\s*([0-9.]+)'
                )
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    try:
                        confidence = float(match)
                        if confidence > 0.2:
                            domain_matches.append(
                                {"domain": domain, "confidence": confidence}
                            )
                    except ValueError:
                        continue

            # Sort by confidence
            domain_matches.sort(key=lambda x: x["confidence"], reverse=True)
            return domain_matches[:3]  # Return top 3 matches

        except Exception:
            return []

    def _classify_by_keywords(self, entry: Dict) -> str:
        """Fallback classification based on content keywords."""
        content = f"{entry.get('instruction', '')} {entry.get('response', '')}".lower()

        # Keyword mapping for each domain
        keyword_mapping = {
            "vulnerability_management": [
                "cve",
                "vulnerability",
                "exploit",
                "patch",
                "security update",
            ],
            "web_security": [
                "web",
                "http",
                "https",
                "xss",
                "csrf",
                "sql injection",
                "owasp",
            ],
            "malware": [
                "malware",
                "virus",
                "trojan",
                "ransomware",
                "backdoor",
                "payload",
            ],
            "phishing": [
                "phishing",
                "email",
                "spoofing",
                "social engineering",
            ],
            "network_security": [
                "network",
                "firewall",
                "intrusion",
                "ddos",
                "packet",
            ],
            "cryptography": [
                "encryption",
                "cryptographic",
                "hash",
                "cipher",
                "rsa",
                "aes",
            ],
            "cloud_security": [
                "cloud",
                "aws",
                "azure",
                "gcp",
                "container",
                "kubernetes",
            ],
            "iot_security": [
                "iot",
                "device",
                "sensor",
                "embedded",
                "firmware",
            ],
            "zero_day": [
                "zero-day",
                "0day",
                "unknown vulnerability",
                "undisclosed",
            ],
            "identity_access_management": [
                "authentication",
                "authorization",
                "identity",
                "access control",
                "iam",
            ],
        }

        # Count keyword matches for each domain
        domain_scores = {}
        for domain, keywords in keyword_mapping.items():
            score = sum(1 for keyword in keywords if keyword in content)
            if score > 0:
                domain_scores[domain] = score

        # Return the domain with highest score, or default to vulnerability_management
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        else:
            return "vulnerability_management"  # Default fallback

    def _update_stats(self, success: bool, domain: str = None):
        """Thread-safe statistics update."""
        with self._stats_lock:
            self._stats["total_processed"] += 1
            if success:
                self._stats["successful"] += 1
                if domain:
                    self._stats["domain_counts"][domain] = (
                        self._stats["domain_counts"].get(domain, 0) + 1
                    )
            else:
                self._stats["failed"] += 1

    def _classify_entries_batch(
        self, entries: List[Dict], batch_id: int, output_file: Path = None
    ) -> List[Dict]:
        """Classify a batch of entries in parallel with live writing."""
        logger.info(f"Processing batch {batch_id} with {len(entries)} entries")
        classified_entries = []

        for i, entry in enumerate(entries):
            try:
                classified_entry = self.classify_entry(entry)
                classified_entries.append(classified_entry)

                # Write immediately to output file if provided
                if output_file:
                    self._append_classified_entry_to_json(
                        output_file, classified_entry
                    )

                # Update stats
                primary_domain = classified_entry.get(
                    "primary_domain", "uncategorized"
                )
                self._update_stats(True, primary_domain)

                if (i + 1) % 10 == 0:  # Log progress every 10 entries
                    logger.info(
                        f"Batch {batch_id}: Processed {i + 1}/{len(entries)} entries"
                    )

            except Exception as e:
                logger.error(
                    f"Error classifying entry in batch {batch_id}: {str(e)}"
                )
                # Add entry with default classification
                entry["domains"] = []
                entry["primary_domain"] = "uncategorized"
                classified_entries.append(entry)
                self._update_stats(False)

                # Write failed entry to output file
                if output_file:
                    self._append_classified_entry_to_json(output_file, entry)

        return classified_entries

    def _append_classified_entry_to_json(self, file_path: Path, entry: Dict):
        """Thread-safe append a single classified entry to the JSON file."""
        with self._file_lock:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Remove trailing ']}' if it exists
            if content.endswith("]}"):
                content = content[:-2]

            # Add comma if not first entry
            if '"data": [' in content and not content.endswith("["):
                content += ",\n" + json.dumps(
                    entry, ensure_ascii=False, indent=2
                )
            else:
                content += "\n" + json.dumps(
                    entry, ensure_ascii=False, indent=2
                )

            # Close the structure
            content += "\n]}"

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

    def _update_metadata(
        self, file_path: Path, processed_files: int, total_entries: int
    ):
        """Update metadata fields in the JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Remove trailing ']}' if it exists
        if content.endswith("]}"):
            content = content[:-2]

        # Update metadata fields
        content = re.sub(
            r'"total_entries": \d+',
            f'"total_entries": {total_entries}',
            content,
        )
        content = re.sub(
            r'"processed_files": \d+',
            f'"processed_files": {processed_files}',
            content,
        )

        # Close the structure
        content += "\n]}"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    def process_directory(self):
        """Process all files in the input directory with live writing."""
        input_files = list(self.input_dir.glob("*.json"))
        if not input_files:
            logger.warning(f"No JSON files found in {self.input_dir}")
            return

        logger.info(f"Found {len(input_files)} files to process")

        # Create consolidated output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        consolidated_output_path = (
            self.output_dir
            / f"consolidated_classified_dataset_{timestamp}.json"
        )

        # Initialize consolidated file with metadata
        initial_metadata = {
            "metadata": {
                "total_entries": 0,
                "processed_files": 0,
                "classification_timestamp": timestamp,
                "model_used": self.ollama_model,
            },
            "data": [],
        }

        with open(consolidated_output_path, "w", encoding="utf-8") as f:
            json.dump(initial_metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Output will be saved to: {consolidated_output_path}")

        # Signal handler for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(
                f"\nReceived interrupt signal. Processed {processed_files}/{len(input_files)} files"
            )
            # Update final metadata
            self._update_metadata(
                consolidated_output_path, processed_files, total_entries
            )
            logger.info(
                f"Classified file saved with current progress: {consolidated_output_path}"
            )
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        processed_files = 0
        total_entries = 0

        # Process files in parallel with live writing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {}

            for file_path in input_files:
                future = executor.submit(
                    self._process_single_file_with_live_writing,
                    file_path,
                    consolidated_output_path,
                )
                future_to_file[future] = file_path

            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    entries_processed = future.result()
                    if entries_processed > 0:
                        processed_files += 1
                        total_entries += entries_processed
                        logger.info(
                            f"✅ Completed processing {file_path} ({entries_processed} entries)"
                        )

                        # Update metadata after each file
                        self._update_metadata(
                            consolidated_output_path,
                            processed_files,
                            total_entries,
                        )
                    else:
                        logger.warning(f"⚠️ No data processed from {file_path}")
                except Exception as e:
                    logger.error(f"❌ Error processing {file_path}: {str(e)}")

        # Final metadata update
        self._update_metadata(
            consolidated_output_path, processed_files, total_entries
        )
        logger.info(f"Final classified file: {consolidated_output_path}")

        # Log final statistics
        self._log_final_statistics(processed_files, total_entries)

    def _process_single_file_with_live_writing(
        self, file_path: Path, output_file: Path
    ) -> int:
        """Process a single file with live writing to output file."""
        logger.info(f"Processing {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict) or "data" not in data:
                logger.warning(f"Invalid data format in {file_path}")
                return 0

            total_entries = len(data["data"])
            logger.info(f"Processing {total_entries} entries from {file_path}")

            # Split entries into batches for parallel processing
            batch_size = max(1, total_entries // self.max_workers)
            batches = [
                data["data"][i : i + batch_size]
                for i in range(0, total_entries, batch_size)
            ]

            entries_processed = 0

            # Process batches in parallel with live writing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                batch_futures = []
                for i, batch in enumerate(batches):
                    future = executor.submit(
                        self._classify_entries_batch, batch, i, output_file
                    )
                    batch_futures.append(future)

                # Collect results from all batches
                for future in as_completed(batch_futures):
                    try:
                        batch_results = future.result()
                        entries_processed += len(batch_results)
                    except Exception as e:
                        logger.error(f"Error processing batch: {str(e)}")

            logger.info(
                f"Completed {file_path}: {entries_processed} entries processed"
            )
            return entries_processed

        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return 0

    def _process_single_file(self, file_path: Path) -> List[Dict]:
        """Process a single file and return classified entries."""
        logger.info(f"Processing {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict) or "data" not in data:
                logger.warning(f"Invalid data format in {file_path}")
                return []

            total_entries = len(data["data"])
            logger.info(f"Processing {total_entries} entries from {file_path}")

            # Split entries into batches for parallel processing
            batch_size = max(1, total_entries // self.max_workers)
            batches = [
                data["data"][i : i + batch_size]
                for i in range(0, total_entries, batch_size)
            ]

            classified_entries = []

            # Process batches in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                batch_futures = []
                for i, batch in enumerate(batches):
                    future = executor.submit(
                        self._classify_entries_batch, batch, i
                    )
                    batch_futures.append(future)

                # Collect results from all batches
                for future in as_completed(batch_futures):
                    try:
                        batch_results = future.result()
                        classified_entries.extend(batch_results)
                    except Exception as e:
                        logger.error(f"Error processing batch: {str(e)}")

            # Update the data structure
            data["data"] = classified_entries
            data["metadata"]["classification_timestamp"] = (
                datetime.now().strftime("%Y%m%d_%H%M%S")
            )

            # Save classified data
            output_file = self.output_dir / f"{file_path.stem}_classified.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved classified data to {output_file}")
            return classified_entries

        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return []

    def _log_final_statistics(self, processed_files: int, total_entries: int):
        """Log final processing statistics."""
        logger.info("\n" + "=" * 60)
        logger.info("CLASSIFICATION STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Processed Files: {processed_files}")
        logger.info(f"Total Entries: {total_entries}")

        with self._stats_lock:
            logger.info(
                f"Successful Classifications: {self._stats['successful']}"
            )
            logger.info(f"Failed Classifications: {self._stats['failed']}")

            if self._stats["domain_counts"]:
                logger.info("\nDomain Distribution:")
                total_classified = sum(self._stats["domain_counts"].values())
                for domain, count in sorted(
                    self._stats["domain_counts"].items()
                ):
                    percentage = (
                        (count / total_classified) * 100
                        if total_classified > 0
                        else 0
                    )
                    logger.info(
                        f"  {domain}: {count} entries ({percentage:.1f}%)"
                    )

        logger.info("=" * 60)


def main():
    """Main function to demonstrate usage."""
    try:
        # Configurable parameters
        classifier = CyberDomainClassifier(
            input_dir="structured_data",
            output_dir="domain_classified",
            ollama_model="gemma3",
            ollama_port=11434,
            max_workers=60,  # Adjust based on your system capabilities
        )
        classifier.process_directory()
    except ConnectionError as e:
        logger.error(f"Connection error: {e}")
        logger.error("Please ensure Ollama is running and accessible")
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
