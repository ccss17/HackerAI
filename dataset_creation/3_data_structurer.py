#!/usr/bin/env python3

import re
import json
import logging
import pandas as pd
import yaml
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import re
import sys
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CyberDataStructurer:
    def __init__(
        self,
        input_dir: str = "filtered_data",
        output_dir: str = "structured_data",
        ollama_model: str = "gemma3",
        ollama_port: int = 65008,
        max_workers: int = 2,
    ):
        """Initialize the data structurer with directory configurations."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ollama_model = ollama_model
        self.ollama_url = f"http://localhost:{ollama_port}/api/generate"
        self.max_workers = max(1, max_workers)
        self._file_lock = (
            threading.Lock()
        )  # Single lock for the consolidated file
        self.wrapper_keys = {
            "results",
            "papers",
            "ctftime_events",
            "vulnerabilities",
            "exploits",
            "data",
            "challenges",
            "samples",
            "phishing_urls",
            "objects",
            "advisories",
            "issues",
            "summary",
            "detailed",
            "links",
            "value",
            "enhanced",
            "affected_packages",
            "securityVulnerabilities",
        }
        self.metadata_keys = {
            "count",
            "prefetch_pulse_ids",
            "t",
            "t2",
            "t3",
            "previous",
            "next",
            "timestamp",
            "status",
            "note",
            "resultsPerPage",
            "startIndex",
            "totalResults",
            "format",
            "version",
            "@odata.context",
            "filtered_reason",
            "query_status",
            "source",
            "enhanced_text",
            "metadata",
        }

        # Verify Ollama connection
        if not self._verify_ollama_connection():
            logger.error(f"Could not connect to Ollama at {self.ollama_url}")
            raise ConnectionError(
                f"Could not connect to Ollama at {self.ollama_url}"
            )

        # Template patterns for different data types
        self.templates = {
            "vulnerability": {
                "system_prompt": """You are a cybersecurity expert. Analyze the following vulnerability data and create a detailed, technical response.
                Focus on technical details, impact, and mitigation strategies. Format your response in a clear, structured way.""",
                "instruction": [
                    "Explain the {cve_id} vulnerability and its potential impact.",
                    "What are the security implications of {cve_id}?",
                    "Describe the vulnerability identified as {cve_id}.",
                    "How does the {cve_id} vulnerability affect system security?",
                ],
            },
            "attack_pattern": {
                "system_prompt": """You are a cybersecurity expert. Analyze the following attack pattern data and create a detailed, technical response.
                Focus on attack methodology, techniques used, and defense strategies. Format your response in a clear, structured way.""",
                "instruction": [
                    "How does the {attack_name} attack work?",
                    "Explain the methodology of {attack_name}.",
                    "What is the {attack_name} attack pattern?",
                    "Describe the execution of {attack_name} attack.",
                ],
            },
            "security_advisory": {
                "system_prompt": """You are a cybersecurity expert. Analyze the following security advisory data and create a detailed, technical response.
                Focus on the security implications, recommended actions, and implementation details. Format your response in a clear, structured way.""",
                "instruction": [
                    "What are the recommended actions for {advisory_id}?",
                    "Explain the security advisory {advisory_id}.",
                    "What measures should be taken regarding {advisory_id}?",
                    "Describe the security implications and fixes for {advisory_id}.",
                ],
            },
            "threat_report": {
                "system_prompt": """You are a threat intelligence analyst. Review the threat report and deliver concise, actionable insights for defenders.""",
                "instruction": [
                    "Summarize the threat campaign described in {report_name}, covering adversary behavior, malware, and targets.",
                    "What defensive actions and mitigations are recommended in the report {report_name}?",
                ],
            },
            "research_paper": {
                "system_prompt": """You are a cybersecurity researcher. Analyze the academic paper and explain its key contributions and relevance.""",
                "instruction": [
                    "Explain the core contribution of the research paper titled {paper_title}.",
                    "What cybersecurity problem does {paper_title} address and what solution does it propose?",
                ],
            },
            "ctf_event": {
                "system_prompt": """You are a cybersecurity training coordinator. Review the CTF event information and extract key details for participants.""",
                "instruction": [
                    "Provide an overview of the CTF event {event_name}, including timeline, format, and difficulty.",
                    "What skills or topics are highlighted during the CTF event {event_name}?",
                ],
            },
            "exploit": {
                "system_prompt": """You are a cybersecurity expert. Analyze the following exploit entry and create a detailed, technical response.
                Focus on exploitation technique, affected components, and mitigations. Format your response in a clear, structured way.""",
                "instruction": [
                    "Explain the exploitation described by {attack_name} and its potential impact.",
                    "How does the {attack_name} exploit work and what defenses mitigate it?",
                ],
            },
        }

    def _verify_ollama_connection(self) -> bool:
        """Verify connection to Ollama API."""
        try:
            response = requests.get(
                self.ollama_url.replace("/generate", "/version"), timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error connecting to Ollama: {str(e)}")
            return False

    def call_ollama(
        self, prompt: str, system_prompt: str, max_retries: int = 3
    ) -> Optional[str]:
        """Call Ollama API with retry mechanism."""
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.ollama_url,
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "system": system_prompt,
                        "stream": False,
                        "keep_alive": "30m",
                        "timeout": 120,
                        "options": {
                            "temperature": 0.2,
                            "num_predict": 1024,  # Limit response length
                            "num_ctx": 8192,  # Limit response length
                            "top_p": 0.9,
                            "repeat_penalty": 1.1,
                        },
                    },
                    timeout=120,  # Reduced timeout
                )
                response.raise_for_status()
                return response.json()["response"]
            except requests.exceptions.Timeout:
                logger.warning(
                    f"Attempt {attempt + 1} timed out after 120 seconds"
                )
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            except Exception as e:
                logger.warning(
                    f"Unexpected error in attempt {attempt + 1}: {str(e)}"
                )

            if attempt < max_retries - 1:
                time.sleep(1)  # Shorter backoff
                continue
        return None

    def load_data(self, file_path: Path) -> Union[Dict, List, None]:
        """Load data from various file formats."""
        try:
            suffix = file_path.suffix.lower()
            if suffix == ".json":
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            elif suffix in {".yaml", ".yml"}:
                with open(file_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
            elif suffix == ".csv":
                return pd.read_csv(file_path).to_dict("records")
            else:
                logger.warning(f"Unsupported file format: {suffix}")
                return None
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            return None

    def detect_entry_type(self, entry: Dict) -> Optional[str]:
        """Detect the type of cybersecurity data entry."""
        # Check for CVE pattern
        if any(key.lower().startswith("cve-") for key in entry.keys()) or any(
            "cve-" in str(value).lower() for value in entry.values()
        ):
            return "vulnerability"

        entry_id = str(entry.get("id", "")).lower()
        if "arxiv.org" in entry_id or "arxiv_primary_category" in entry:
            return "research_paper"

        # Check for attack pattern indicators
        if any(
            key in entry for key in {"attack_pattern", "technique", "tactic"}
        ) or any("attack" in str(value).lower() for value in entry.values()):
            return "attack_pattern"

        # Check for security advisory
        if any(
            key in entry for key in {"advisory", "bulletin", "notice"}
        ) or any(
            isinstance(value, str) and "advisory" in value.lower()
            for value in entry.values()
        ):
            return "security_advisory"

        # Check for threat intelligence reports (OTX style)
        if any(
            key in entry
            for key in {
                "indicators",
                "malware_families",
                "tlp",
                "attack_ids",
                "industries",
            }
        ):
            return "threat_report"

        # Check for CTF event listings
        if any(
            key in entry
            for key in {
                "ctf_id",
                "ctftime_url",
                "start",
                "finish",
                "format",
                "duration",
            }
        ):
            return "ctf_event"

        # Check for exploit entries (exploit-db style)
        if ("file" in entry and "description" in entry) or (
            "type" in entry
            and entry.get("type") in {"local", "remote", "webapps", "dos"}
        ):
            return "exploit"

        return None

    def extract_fields(self, entry: Dict, entry_type: str) -> Dict:
        """Extract relevant fields based on entry type."""
        fields = {}

        if entry_type == "vulnerability":
            # Extract CVE ID
            cve_pattern = r"CVE-\d{4}-\d{4,7}"
            cve_matches = []
            for value in entry.values():
                if isinstance(value, str):
                    matches = re.findall(cve_pattern, value, re.IGNORECASE)
                    cve_matches.extend(matches)

            fields["cve_id"] = cve_matches[0] if cve_matches else "Unknown CVE"
            fields["raw_data"] = json.dumps(entry, indent=2)

        elif entry_type == "attack_pattern":
            fields["attack_name"] = self._find_field(
                entry, ["name", "title", "pattern_name"]
            )
            fields["raw_data"] = json.dumps(entry, indent=2)

        elif entry_type == "security_advisory":
            fields["advisory_id"] = self._find_field(
                entry, ["id", "advisory_id", "bulletin_id"]
            )
            fields["raw_data"] = json.dumps(entry, indent=2)

        elif entry_type == "threat_report":
            fields["report_name"] = self._find_field(entry, ["name", "title"])
            fields["primary_malware"] = (
                self._stringify(entry.get("malware_families"))
                or "Not specified"
            )
            fields["raw_data"] = json.dumps(entry, indent=2)

        elif entry_type == "research_paper":
            fields["paper_title"] = self._find_field(entry, ["title"])
            fields["published_date"] = self._find_field(
                entry, ["published", "updated"]
            )
            fields["raw_data"] = json.dumps(entry, indent=2)

        elif entry_type == "ctf_event":
            fields["event_name"] = self._find_field(entry, ["title", "name"])
            fields["event_start"] = (
                self._stringify(entry.get("start")) or "Not provided"
            )
            fields["event_format"] = self._find_field(entry, ["format"])
            fields["raw_data"] = json.dumps(entry, indent=2)

        elif entry_type == "exploit":
            # Use description as the primary name for exploits
            fields["attack_name"] = self._find_field(
                entry, ["description", "file", "id"]
            )
            fields["exploit_id"] = self._find_field(entry, ["id"]) or "Unknown"
            fields["platform"] = self._find_field(entry, ["platform", "type"])
            fields["raw_data"] = json.dumps(entry, indent=2)

        return fields

    def _find_field(self, entry: Dict, possible_keys: List[str]) -> str:
        """Helper method to find a field value from possible keys."""
        for key in possible_keys:
            for entry_key, value in entry.items():
                if key.lower() in entry_key.lower():
                    if isinstance(value, str):
                        return value
                    if isinstance(value, list):
                        string_values = [
                            item for item in value if isinstance(item, str)
                        ]
                        if string_values:
                            return ", ".join(string_values)
                    if value not in (None, ""):
                        return str(value)
        return "Information not available"

    def _stringify(self, value: Union[str, List, Dict, None]) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            string_values = [self._stringify(item) for item in value]
            return ", ".join(filter(None, string_values))
        if isinstance(value, dict):
            return json.dumps(value, indent=2)
        return str(value)

    def _looks_like_record(self, entry: Dict) -> bool:
        """Heuristic check to identify entries containing substantive content."""
        if not isinstance(entry, dict) or not entry:
            return False

        normalized_keys = {key.lower() for key in entry.keys()}
        link_like_keys = {"href", "rel", "type", "title", "length"}
        if normalized_keys and normalized_keys.issubset(link_like_keys):
            return False

        indicative_keys = {
            "id",
            "title",
            "name",
            "summary",
            "description",
            "cve",
            "cve_id",
            "attack_name",
            "advisory_id",
            "report_name",
            "paper_title",
        }

        if any(key in entry for key in indicative_keys):
            return True

        # Some feeds store the primary content under "content" or "data" fields
        if any(key in entry for key in {"content", "data"}):
            return True

        return False

    def _extract_candidate_entries(
        self, entry: Dict, *, allow_nested_wrapper: bool = False
    ) -> List[Dict]:
        """Extract actionable record dictionaries from possibly wrapped entries."""
        if not isinstance(entry, dict):
            return []

        candidates: List[Dict] = []
        seen_signatures = set()

        def normalise(candidate: Dict) -> Optional[Dict]:
            if not isinstance(candidate, dict):
                return None

            filtered = {
                key: value
                for key, value in candidate.items()
                if key not in self.metadata_keys
            }

            if not filtered or not self._looks_like_record(filtered):
                return None

            signature = (
                filtered.get("id")
                or filtered.get("name")
                or filtered.get("title")
            )
            if isinstance(signature, (dict, list)):
                signature = json.dumps(signature, sort_keys=True)

            if signature:
                sig = ("sig", signature)
            else:
                sig = ("hash", json.dumps(filtered, sort_keys=True))

            if sig in seen_signatures:
                return None

            seen_signatures.add(sig)
            return filtered

        initial = normalise(entry)
        if initial:
            candidates.append(initial)

        for key, value in entry.items():
            if key not in self.wrapper_keys:
                continue

            if isinstance(value, list):
                for item in value:
                    normalised = normalise(item)
                    if normalised:
                        candidates.append(normalised)
                    if allow_nested_wrapper and isinstance(item, dict):
                        candidates.extend(
                            self._extract_candidate_entries(
                                item, allow_nested_wrapper=False
                            )
                        )
            elif isinstance(value, dict):
                normalised = normalise(value)
                if normalised:
                    candidates.append(normalised)
                if allow_nested_wrapper:
                    candidates.extend(
                        self._extract_candidate_entries(
                            value, allow_nested_wrapper=False
                        )
                    )

        return candidates

    def create_instruction_response_pair(
        self, entry: Dict, entry_type: str, output_file: Path = None
    ) -> List[Dict]:
        """Create instruction-response pairs from an entry using Ollama."""
        fields = self.extract_fields(entry, entry_type)
        template = self.templates[entry_type]
        pairs = []

        # Only use the first two instruction templates to reduce processing time
        for instruction_template in template["instruction"][:2]:
            instruction = instruction_template.format(**fields)

            # Create a focused prompt for Ollama with just the relevant entry data
            prompt = f"""Analyze this {entry_type} and answer: {instruction}

Data: {json.dumps(entry, indent=2)}
Requirements:
- Provide a comprehensive analysis in 2-3 paragraphs
- Maximum 400 words
- Complete all sentences
- No conversational elements (avoid "Do you want me to...")"""

            response = self.call_ollama(prompt, template["system_prompt"])
            if response:
                pair = {
                    "instruction": instruction,
                    "response": response,
                    "type": entry_type,
                    "source_data": {
                        "id": fields.get(
                            "cve_id",
                            fields.get(
                                "attack_name",
                                fields.get("advisory_id", "Unknown"),
                            ),
                        ),
                        "type": entry_type,
                    },
                }
                pairs.append(pair)

                # Write immediately if output file is provided
                if output_file:
                    self._append_pair_to_json(output_file, pair)

        return pairs

    def structure_dataset(
        self, input_file: Path, output_file: Path = None
    ) -> List[Dict]:
        """Structure the dataset into instruction-response format."""
        data = self.load_data(input_file)
        if not data:
            return []

        # Initialize output file with metadata
        if output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            initial_metadata = {
                "metadata": {
                    "source_file": str(input_file.name),
                    "total_entries": 0,
                    "generation_timestamp": timestamp,
                    "model_used": self.ollama_model,
                },
                "data": [],
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(initial_metadata, f, indent=2, ensure_ascii=False)

        flattened_entries = self.flatten_entries(data)
        structured_pairs = []
        total_entries = len(flattened_entries)

        for i, entry in enumerate(flattened_entries, 1):
            logger.info(
                f"Processing entry {i}/{total_entries} from {input_file.name}"
            )
            if not isinstance(entry, dict):
                logger.warning(
                    f"Skipped non-dict entry at position {i} in {input_file.name}"
                )
                continue

            candidate_entries = self._extract_candidate_entries(
                entry, allow_nested_wrapper=True
            )
            if not candidate_entries:
                logger.debug(
                    f"Skipping entry {i} from {input_file.name}: no actionable content detected"
                )
                continue

            processed = False
            for candidate in candidate_entries:
                entry_type = self.detect_entry_type(candidate)
                if not entry_type:
                    logger.debug(
                        "Candidate from entry %d in %s could not be typed",
                        i,
                        input_file.name,
                    )
                    continue

                pairs = self.create_instruction_response_pair(
                    candidate, entry_type, output_file
                )
                if pairs:
                    structured_pairs.extend(pairs)
                    processed = True

            if not processed:
                # Provide a compact summary of why the entry produced no pairs
                try:
                    candidate_summary = {
                        "candidate_count": len(candidate_entries),
                        "candidate_keys": [
                            list(c.keys()) for c in candidate_entries[:3]
                        ],
                    }
                except Exception:
                    candidate_summary = {"candidate_count": 0}

                logger.warning(
                    f"No instruction-response pairs generated for entry {i}: {candidate_summary}"
                )

        # Update final metadata if output file was provided
        if output_file and structured_pairs:
            self._update_metadata(output_file, 1, len(structured_pairs))

        return structured_pairs

    def process_directory(self):
        """Process all files in the input directory and save to a single consolidated JSON file."""
        processed_files = 0
        total_pairs = 0

        # Get all JSON files in the input directory
        input_files = list(self.input_dir.glob("*_filtered_*.json"))
        total_files = len(input_files)

        if not input_files:
            logger.warning(f"No filtered files found in {self.input_dir}")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        consolidated_output_path = (
            self.output_dir
            / f"consolidated_cybersecurity_dataset_{timestamp}.json"
        )

        logger.info(f"Found {total_files} files to process")
        logger.info(f"Output will be saved to: {consolidated_output_path}")

        # Check for existing consolidated file (resume capability)
        existing_files = list(
            self.output_dir.glob("consolidated_cybersecurity_dataset_*.json")
        )
        if existing_files:
            # Use the most recent existing file for resume
            consolidated_output_path = max(
                existing_files, key=lambda f: f.stat().st_mtime
            )
            logger.info(
                f"Resuming from existing file: {consolidated_output_path}"
            )

            # Load existing data to get current progress
            try:
                with open(
                    consolidated_output_path, "r", encoding="utf-8"
                ) as f:
                    existing_data = json.load(f)
                processed_files = existing_data.get("metadata", {}).get(
                    "processed_files", 0
                )
                total_pairs = existing_data.get("metadata", {}).get(
                    "total_entries", 0
                )
                logger.info(
                    f"Resuming: {processed_files} files already processed, {total_pairs} pairs generated"
                )
            except Exception as e:
                logger.warning(f"Could not load existing file for resume: {e}")
                processed_files = 0
                total_pairs = 0
        else:
            # Initialize new consolidated JSON file with metadata
            initial_metadata = {
                "metadata": {
                    "total_entries": 0,
                    "processed_files": 0,
                    "generation_timestamp": timestamp,
                    "model_used": self.ollama_model,
                },
                "data": [],
            }

            with open(consolidated_output_path, "w", encoding="utf-8") as f:
                json.dump(initial_metadata, f, indent=2, ensure_ascii=False)

        # Set up signal handler for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(
                f"\nReceived interrupt signal. Processed {processed_files}/{total_files} files, {total_pairs} total pairs"
            )
            # Update final metadata
            self._update_metadata(
                consolidated_output_path, processed_files, total_pairs
            )
            logger.info(
                f"Consolidated file saved with current progress: {consolidated_output_path}"
            )
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # Process files in parallel with thread-safe file writing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_info = {}

            for i, input_file in enumerate(input_files, 1):
                logger.info(
                    f"Submitting file {i}/{total_files}: {input_file.name}"
                )

                # All threads write to the same consolidated file (thread-safe)
                future = executor.submit(
                    self.structure_dataset,
                    input_file,
                    consolidated_output_path,
                )
                future_to_info[future] = (input_file, i)

            # Collect results as they complete
            for future in as_completed(future_to_info):
                input_file, index = future_to_info[future]

                try:
                    pairs = future.result()

                    if pairs:
                        processed_files += 1
                        total_pairs += len(pairs)
                        logger.info(
                            f"Generated {len(pairs)} pairs from {input_file.name}"
                        )

                        # Update metadata after each file
                        self._update_metadata(
                            consolidated_output_path,
                            processed_files,
                            total_pairs,
                        )
                    else:
                        logger.warning(
                            f"No pairs generated from {input_file.name}"
                        )

                except Exception as e:
                    logger.error(
                        f"Error processing {input_file.name}: {str(e)}"
                    )
                    continue

        logger.info("Processing Statistics:")
        logger.info(f"Processed Files: {processed_files}")
        logger.info(f"Total Instruction-Response Pairs: {total_pairs}")

        # Final metadata update
        self._update_metadata(
            consolidated_output_path, processed_files, total_pairs
        )
        logger.info(f"Final consolidated file: {consolidated_output_path}")

    def flatten_entries(self, data: Union[Dict, List]) -> List[Dict]:
        """Recursively extract record-like dictionaries from nested structures."""
        collected: List[Dict] = []

        def recurse(node: Union[Dict, List, str, int, float, None]):
            if isinstance(node, list):
                for item in node:
                    recurse(item)
                return

            if isinstance(node, dict):
                keys = [
                    key for key in node.keys() if key not in self.metadata_keys
                ]

                # First, dive into known wrapper keys
                consumed_wrapper = False
                for key in keys:
                    if key in self.wrapper_keys and isinstance(
                        node[key], (list, dict)
                    ):
                        recurse(node[key])
                        consumed_wrapper = True

                # Special-case STIX bundles (type == 'bundle' and objects key)
                if "objects" in node and node.get("type") == "bundle":
                    recurse(node["objects"])
                    consumed_wrapper = True

                remaining_keys = [
                    key for key in keys if key not in self.wrapper_keys
                ]

                if remaining_keys and isinstance(node, dict):
                    record = {key: node[key] for key in remaining_keys}
                    collected.append(record)
                elif not consumed_wrapper and not remaining_keys:
                    # No meaningful keys left; keep original dict to avoid losing data
                    collected.append(node)
                return

            # Primitive node: wrap as dict so caller can decide how to use it
            collected.append({"value": node})

        recurse(data)
        return collected

    def _append_pair_to_json(self, file_path: Path, pair: Dict):
        """Thread-safe append a single pair to the JSON file."""
        # Thread-safe file writing with single lock
        with self._file_lock:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Remove trailing ']}' if it exists
            if content.endswith("]}"):
                content = content[:-2]

            # Add comma if not first entry
            if '"data": [' in content and not content.endswith("["):
                content += ",\n" + json.dumps(
                    pair, ensure_ascii=False, indent=2
                )
            else:
                content += "\n" + json.dumps(
                    pair, ensure_ascii=False, indent=2
                )

            # Close the structure
            content += "\n]}"

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

    def _update_metadata(
        self, file_path: Path, processed_files: int, total_pairs: int
    ):
        """Update metadata fields in the JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Remove trailing ']}' if it exists
        if content.endswith("]}"):
            content = content[:-2]

        content = re.sub(
            r'"total_entries": \d+', f'"total_entries": {total_pairs}', content
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


def main():
    """Main function to demonstrate usage."""
    try:
        structurer = CyberDataStructurer(
            ollama_port=11434, max_workers=30
        )  # Use the default port
        structurer.process_directory()
    except ConnectionError as e:
        logger.error(f"Failed to initialize: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
