#!/usr/bin/env python3

import json
import logging
import requests
import time
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
)
from rich.console import Console

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatasetPreprocessor:
    def __init__(
        self,
        input_file: str = "final_dataset/final_cybersecurity_dataset_20251013_022503.json",
        output_dir: str = "final_dataset_preprocessed",
        ollama_model: str = "gemma3:latest",
        ollama_port: int = 11434,
        max_workers: int = 4,
    ):
        """Initialize the dataset preprocessor."""
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ollama_model = ollama_model
        self.ollama_url = f"http://localhost:{ollama_port}/api/generate"
        self.max_workers = max(1, max_workers)
        self.console = Console()

        # Thread-safe file writing
        self._file_lock = threading.Lock()
        self._output_file = None

        # Statistics tracking
        self.stats = {
            "total_processed": 0,
            "successful_regenerations": 0,
            "duplicates_removed": 0,
            "errors": 0,
            "responses_cleaned": 0,
        }

        # Track seen instructions for deduplication (thread-safe)
        self._seen_instructions_lock = threading.Lock()
        self.seen_instructions: Set[str] = set()

        # Verify Ollama connection
        if not self._verify_ollama_connection():
            logger.error(f"Could not connect to Ollama at {self.ollama_url}")
            raise ConnectionError(
                f"Could not connect to Ollama at {self.ollama_url}"
            )

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

    def call_ollama(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Call Ollama API with retry mechanism."""
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.ollama_url,
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "keep_alive": "30m",
                        "timeout": 120,
                        "options": {
                            "temperature": 0.3,
                            "num_predict": 200,
                            "num_ctx": 4096,
                            "top_p": 0.9,
                            "repeat_penalty": 1.1,
                        },
                    },
                    timeout=120,
                )
                response.raise_for_status()
                return response.json()["response"].strip()
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
                time.sleep(2)  # Backoff between retries
                continue
        return None

    def clean_response(self, response: str) -> str:
        """Remove '##' markdown headers from response start."""
        if not response:
            return response

        # Remove leading '##' and any following whitespace/newlines
        cleaned = re.sub(r"^##\s*", "", response.strip())

        # Also remove any other markdown headers at the start
        cleaned = re.sub(r"^#{1,6}\s*", "", cleaned)

        return cleaned

    def _initialize_output_file(self) -> Path:
        """Initialize the output file with metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._output_file = (
            self.output_dir
            / f"final_cybersecurity_dataset_preprocessed_{timestamp}.json"
        )

        # Create initial metadata
        initial_metadata = {
            "metadata": {
                "source_file": str(self.input_file.name),
                "total_entries": 0,
                "processing_timestamp": timestamp,
                "model_used": self.ollama_model,
                "statistics": self.stats,
            },
            "data": [],
        }

        with open(self._output_file, "w", encoding="utf-8") as f:
            json.dump(initial_metadata, f, indent=2, ensure_ascii=False)

        return self._output_file

    def _append_entry_to_file(self, entry: Dict):
        """Thread-safe append a single entry to the JSON file."""
        with self._file_lock:
            with open(self._output_file, "r", encoding="utf-8") as f:
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

            with open(self._output_file, "w", encoding="utf-8") as f:
                f.write(content)

    def _update_metadata(self, total_entries: int, processed_entries: int):
        """Update metadata fields in the JSON file."""
        with self._file_lock:
            with open(self._output_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Remove trailing ']}' if it exists
            if content.endswith("]}"):
                content = content[:-2]

            # Update statistics
            content = re.sub(
                r'"total_entries": \d+',
                f'"total_entries": {total_entries}',
                content,
            )
            content = re.sub(
                r'"total_processed": \d+',
                f'"total_processed": {self.stats["total_processed"]}',
                content,
            )
            content = re.sub(
                r'"successful_regenerations": \d+',
                f'"successful_regenerations": {self.stats["successful_regenerations"]}',
                content,
            )
            content = re.sub(
                r'"duplicates_removed": \d+',
                f'"duplicates_removed": {self.stats["duplicates_removed"]}',
                content,
            )
            content = re.sub(
                r'"responses_cleaned": \d+',
                f'"responses_cleaned": {self.stats["responses_cleaned"]}',
                content,
            )
            content = re.sub(
                r'"errors": \d+',
                f'"errors": {self.stats["errors"]}',
                content,
            )

            # Close the structure
            content += "\n]}"

            with open(self._output_file, "w", encoding="utf-8") as f:
                f.write(content)

    def generate_instruction_from_response(
        self, response: str
    ) -> Optional[str]:
        """Generate a new instruction based on the response content using Ollama."""
        if not response:
            return None

        # Clean the response first
        cleaned_response = self.clean_response(response)

        prompt = f"""Analyze this cybersecurity response and generate a clear, specific instruction/question that this response would answer.

Response: {cleaned_response}

Generate ONLY the instruction text, no additional commentary. The instruction should be:
- Specific and technical
- Related to the actual content
- In question or directive form
- Professional and clear"""

        return self.call_ollama(prompt)

    def process_entry(self, entry: Dict) -> Optional[Dict]:
        """Process a single entry: regenerate instruction, clean response, check for duplicates."""
        if not isinstance(entry, dict):
            return None

        original_instruction = entry.get("instruction", "")
        original_response = entry.get("response", "")

        # Thread-safe duplicate check
        with self._seen_instructions_lock:
            if original_instruction in self.seen_instructions:
                self.stats["duplicates_removed"] += 1
                return None

        # Generate new instruction from response
        new_instruction = self.generate_instruction_from_response(
            original_response
        )
        if not new_instruction:
            logger.warning("Failed to generate instruction for entry")
            self.stats["errors"] += 1
            return None

        # Clean the response (remove ## headers)
        cleaned_response = self.clean_response(original_response)
        if cleaned_response != original_response:
            self.stats["responses_cleaned"] += 1

        # Create processed entry
        processed_entry = {
            "instruction": new_instruction,
            "response": cleaned_response,
        }

        # Preserve metadata if it exists
        if "metadata" in entry:
            processed_entry["metadata"] = entry["metadata"]
        if "source_data" in entry:
            processed_entry["source_data"] = entry["source_data"]
        if "type" in entry:
            processed_entry["type"] = entry["type"]

        # Thread-safe duplicate tracking
        with self._seen_instructions_lock:
            self.seen_instructions.add(new_instruction)
            self.stats["successful_regenerations"] += 1

        # Write immediately to file
        self._append_entry_to_file(processed_entry)

        return processed_entry

    def load_dataset(self) -> List[Dict]:
        """Load the dataset from JSON file."""
        try:
            with open(self.input_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle both direct lists and data field
            if isinstance(data, dict) and "data" in data:
                return data["data"]
            elif isinstance(data, list):
                return data
            else:
                logger.error("Unexpected data format in input file")
                return []
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return []

    def save_processed_dataset(self, processed_entries: List[Dict]) -> Path:
        """Save the processed dataset to a new file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = (
            self.output_dir
            / f"final_cybersecurity_dataset_preprocessed_{timestamp}.json"
        )

        # Create metadata
        metadata = {
            "metadata": {
                "source_file": str(self.input_file.name),
                "total_entries": len(processed_entries),
                "processing_timestamp": timestamp,
                "model_used": self.ollama_model,
                "statistics": self.stats,
            },
            "data": processed_entries,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return output_file

    def process_dataset(self, test_mode: bool = False):
        """Process the entire dataset with multi-worker processing."""
        logger.info(f"Loading dataset from {self.input_file}")
        entries = self.load_dataset()

        if not entries:
            logger.error("No entries found in dataset")
            return

        total_entries = len(entries)
        if test_mode:
            entries = entries[:20]  # Test with first 20 entries
            logger.info(
                f"TEST MODE: Processing only first {len(entries)} entries"
            )
        else:
            logger.info(f"Processing {total_entries} entries")

        # Initialize output file
        output_file = self._initialize_output_file()
        logger.info(f"Output will be saved to: {output_file}")

        # Set up signal handler for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(
                f"\nReceived interrupt signal. Processed {self.stats['total_processed']} entries"
            )
            logger.info(f"Results saved to: {output_file}")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # Process entries with multi-worker threading
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_entry = {}

            # Submit all entries for processing
            for i, entry in enumerate(entries):
                future = executor.submit(self.process_entry, entry)
                future_to_entry[future] = (entry, i)

            # Process results as they complete
            processed_count = 0
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task(
                    "[cyan]Processing entries...",
                    total=len(entries),
                )

                for future in as_completed(future_to_entry):
                    entry, index = future_to_entry[future]

                    try:
                        processed_entry = future.result()
                        if processed_entry:
                            processed_count += 1

                        self.stats["total_processed"] += 1

                        # Update progress every 50 entries
                        if self.stats["total_processed"] % 50 == 0:
                            progress.update(task, advance=50)
                            # Update metadata periodically
                            self._update_metadata(
                                total_entries, processed_count
                            )

                    except Exception as e:
                        logger.error(
                            f"Error processing entry {index}: {str(e)}"
                        )
                        self.stats["errors"] += 1
                        continue

                # Final progress update
                progress.update(task, completed=len(entries))

        # Final metadata update
        self._update_metadata(total_entries, processed_count)

        # Print statistics
        self.console.print("\n[bold green]Processing Complete![/bold green]")
        self.console.print(
            f"[bold]Total entries processed:[/bold] {self.stats['total_processed']}"
        )
        self.console.print(
            f"[bold]Successful regenerations:[/bold] {self.stats['successful_regenerations']}"
        )
        self.console.print(
            f"[bold]Duplicates removed:[/bold] {self.stats['duplicates_removed']}"
        )
        self.console.print(
            f"[bold]Responses cleaned:[/bold] {self.stats['responses_cleaned']}"
        )
        self.console.print(f"[bold]Errors:[/bold] {self.stats['errors']}")
        self.console.print(
            f"[bold]Final dataset size:[/bold] {processed_count}"
        )
        self.console.print(f"[bold]Output file:[/bold] {output_file}")


def main():
    """Main function to run the preprocessor."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess cybersecurity dataset"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="final_dataset/final_cybersecurity_dataset_20251013_022503.json",
        help="Input dataset file path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="final_dataset_preprocessed",
        help="Output directory for processed dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma3:latest",
        help="Ollama model to use",
    )
    parser.add_argument(
        "--port", type=int, default=11434, help="Ollama server port"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=60,
        help="Number of worker threads (default: 4)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process only first 20 entries",
    )

    args = parser.parse_args()

    try:
        preprocessor = DatasetPreprocessor(
            input_file=args.input_file,
            output_dir=args.output_dir,
            ollama_model=args.model,
            ollama_port=args.port,
            max_workers=args.workers,
        )
        preprocessor.process_dataset(test_mode=args.test)
    except ConnectionError as e:
        logger.error(f"Failed to initialize: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
