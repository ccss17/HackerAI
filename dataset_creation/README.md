# Dataset Creation Pipeline

This directory contains a series of Python scripts that form a comprehensive pipeline for creating, processing, and validating a cyber security dataset for the CyberLLMInstruct paper ([arXiv:2503.09334](https://arxiv.org/abs/2503.09334)). Each script performs a specific function in the pipeline, designed to be run sequentially.

## Prerequisites

1. **Python Environment Setup**
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Optional: Install malwarebazaar library for better MalwareBazaar support
pip install malwarebazaar
```

2. **Ollama Setup**
```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama pull gemma3:latest
```

3. **API Setup (Optional)**
The data collector can fetch data from **many public sources without any API keys**. However, for additional data sources, you can optionally configure API credentials using environment variables.

**Public sources (no API key needed):**
- CVE databases (NVD)
- MITRE ATT&CK & CAPEC
- Ubuntu/Microsoft Security Advisories
- ArXiv papers
- ThreatFox, MalwareBazaar, OpenPhish
- Exploit-DB, Project Zero, ZDI
- CTFtime events

**Optional API credentials (for additional sources):**
```bash
# OpenCVE (requires account at https://app.opencve.io/)
export OPENCVE_EMAIL="your_email@example.com"
export OPENCVE_PASSWORD="your_password"

# AlienVault OTX (get key from https://otx.alienvault.com/)
export ALIENVAULT_API_KEY="your_key_here"

# VirusTotal (get key from https://www.virustotal.com/)
export VIRUSTOTAL_API_KEY="your_key_here"

# HackTheBox (get key from https://www.hackthebox.com/)
export HTB_API_KEY="your_key_here"

# Shodan (get key from https://www.shodan.io/)
export SHODAN_API_KEY="your_key_here"

# ThreatFox (get key from https://threatfox.abuse.ch/)
export THREATFOX_API_KEY="your_key_here"

# MalwareBazaar (get key from https://bazaar.abuse.ch/api/#account)
export MALWAREBAZAAR_API_KEY="your_key_here"

# MalShare (get key from https://malshare.com/)
export MALSHARE_API_KEY="your_key_here"

# Malpedia (INVITE-ONLY - leave unset unless you have access)
export MALPEDIA_API_KEY="your_key_here"

# PhishTank (get key from https://www.phishtank.com/api_register.php)
export PHISHTANK_API_KEY="your_key_here"

# Root-Me (get key from https://www.root-me.org/ account settings)
export ROOTME_API_KEY="your_key_here"

# GitHub (for security advisories)
export GITHUB_TOKEN="your_github_token"
```

**Note:** If API keys are not set, those specific sources will be skipped with a warning, but all public sources will still work.

## Pipeline Overview

1. **Data Collection** (`1_data_collector.py`)
   - Collects raw data from various sources
   - Supports multiple input formats (JSON, CSV, YAML)
   - Handles API rate limiting and error recovery
   - Saves raw data with source metadata
   
   Example output folder:
   ```
   raw_data/
   ├── source1_data_20250507_165224.json
   ├── source2_data_20250507_165224.json
   └── collection_stats.json
   ```

2. **Data Filtering** (`2_data_filter.py`)
   - Removes irrelevant or low-quality entries
   - Applies content filters and quality checks
   - Uses Ollama for content quality assessment
   - Handles duplicate detection
   - Generates filtering statistics
   
   Example output folder:
   ```
   filtered_data/
   ├── filtered_dataset_20250507_165926.json
   └── filtering_stats.json
   ```

3. **Data Structuring** (`3_data_structurer.py`)
   - Standardises data format
   - Validates data structure
   - Uses Ollama for text normalization
   - Ensures consistent metadata fields
   
   Example output folder:
   ```
   structured_data/
   ├── structured_dataset_20250507_165926.json
   └── structure_validation_report.json
   ```

4. **Domain Classification** (`4_domain_classifier.py`)
   - Categorises entries into cyber security domains
   - Uses Ollama for domain classification
   - Supports manual classification corrections
   - Tracks classification confidence scores
   
   Example output folder:
   ```
   classified_data/
   ├── classified_dataset_20250507_170156.json
   └── classification_metrics.json
   ```

5. **Manual Review** (`5_manual_reviewer.py`)
   - Interactive interface for data review
   - Quality assessment tools
   - Uses Ollama for automated quality checks
   - Progress tracking and reporting
   
   Example output folder:
   ```
   reviewed_data/
   ├── reviewed_dataset_20250507_170404.json
   └── review_notes.json
   ```

6. **Security Alignment & Enhancement** (`6_security_aligner.py`)
   - Adds security-focused examples
   - Generates adversarial cases
   - Uses Ollama for instruction-response enhancement
   - Implements compliance testing
   - Flags sensitive content
   - Enhances clarity and coherence
   - Tracks enhancement history
   
   Example output folder:
   ```
   security_aligned/
   ├── consolidated_cybersecurity_dataset_20250507_165224_classified_reviewed_20250507_170156_security_aligned_20250507_172532.json
   └── enhancement_log.json
   ```

7. **Final Assembly** (`8_final_assembler.py`)
   - Merges processed data
   - Performs final validation
   - Removes duplicates
   - Generates clean instruction-response pairs
   - Exports final dataset in JSON format
   
   Example output folder:
   ```
   final_dataset/
   └── final_cybersecurity_dataset_20250507_173441.json
   ```

## Usage

1. Ensure Ollama is running:
```bash
# Start Ollama service
ollama serve
```

2. Run scripts in sequence:
```bash
# 1. Collect data from all available sources (default)
python 1_data_collector.py --output-dir raw_data

# OR collect from specific sources only
python 1_data_collector.py --sources cve_data mitre_attack threatfox --output-dir raw_data

# OR use "all" to explicitly fetch from all sources
python 1_data_collector.py --sources all --output-dir raw_data

# 2. Filter data
python 2_data_filter.py --input-dir raw_data --output-dir filtered_data

# 3. Structure data
python 3_data_structurer.py --input-dir filtered_data --output-dir structured_data

# 4. Classify domains
python 4_domain_classifier.py --input-dir structured_data --output-dir classified_data

# 5. Manual review
python 5_manual_reviewer.py --input-dir classified_data --output-dir reviewed_data

# 6. Security alignment and enhancement
python 6_security_aligner.py --input-dir reviewed_data --output-dir security_aligned

# 7. Final assembly
python 8_final_assembler.py --input-dir security_aligned --output-dir final_dataset
```

## Output Format

The final dataset is available in JSON format with the following structure:
```json
[
  {
    "instruction": "text of the instruction",
    "response": "text of the response"
  }
]
```

Each entry contains only the instruction-response pair, with all metadata and complex structures converted to clean text format.

## Configuration

The pipeline uses the following configuration files:
- `config.json`: General pipeline configuration
- `ollama_config.json`: Ollama API settings and model parameters

Example `ollama_config.json`:
```json
{
  "base_url": "http://localhost:11434",
  "models": {
    "classification": "gemma:2b",
    "enhancement": "mistral:7b"
  },
  "timeout": 60,
  "max_tokens": 2000
}
```

## Available Data Sources

### Public Sources (No API Key Required)
Run the collector without any environment variables - these sources work out of the box:

| Source | Description | Type |
|--------|-------------|------|
| `cve_data` | CVE vulnerabilities from NVD | Vulnerability Database |
| `mitre_attack` | MITRE ATT&CK framework | Attack Patterns |
| `capec_data` | Common Attack Pattern Enumeration | Attack Patterns |
| `ubuntu_security` | Ubuntu Security Notices | Security Advisories |
| `redhat_security` | Red Hat Security Blog | Security Advisories |
| `microsoft_security` | Microsoft Security Updates | Security Advisories |
| `arxiv_papers` | Recent cybersecurity research | Research Papers |
| `openphish` | Phishing URLs feed | Social Engineering |
| `exploit_db` | Recent exploits from Exploit-DB | Exploits |
| `project_zero` | Google Project Zero issues | Vulnerability Research |
| `zerodayinitiative` | Zero Day Initiative advisories | Vulnerability Research |
| `ctf_data` | CTFtime events | Training/CTF |

### API-Required Sources (Optional)
These sources need API credentials set via environment variables:

| Source | Description | Required Env Vars |
|--------|-------------|-------------------|
| `opencve_data` | CVE data from OpenCVE | `OPENCVE_EMAIL`, `OPENCVE_PASSWORD` |
| `alienvault_otx` | AlienVault OTX threat intel | `ALIENVAULT_API_KEY` |
| `threatfox` | Threat indicators from ThreatFox | `THREATFOX_API_KEY` |
| `malware_bazaar` | Malware samples from MalwareBazaar | `MALWAREBAZAAR_API_KEY` |
| `virustotal` | VirusTotal file reports | `VIRUSTOTAL_API_KEY` |
| `malshare` | MalShare malware samples | `MALSHARE_API_KEY` |
| `root_me` | Root-Me challenges | `ROOTME_API_KEY` |
| `hackthebox` | HackTheBox challenges | `HTB_API_KEY` |
| `shodan` | Shodan IoT/network data | `SHODAN_API_KEY` |
| `github_advisories` | GitHub security advisories | `GITHUB_TOKEN` |

### Usage Examples
```bash
# Fetch all public sources (default behavior)
python 1_data_collector.py

# Fetch specific public sources
python 1_data_collector.py --sources cve_data mitre_attack threatfox

# Fetch with API-based sources (after setting environment variables)
export OPENCVE_EMAIL="your@email.com"
export OPENCVE_PASSWORD="password"
python 1_data_collector.py --sources cve_data opencve_data

# Get help and see all available sources
python 1_data_collector.py --help
```

## Notes

- The pipeline requires Ollama to be running locally (for steps 2-6)
- Each script includes progress tracking and error handling
- The final dataset contains only clean instruction-response pairs
- All complex responses are converted to readable text format
- Duplicate entries are automatically removed
- Invalid entries are logged and excluded from the final dataset
- Each stage creates timestamped output files for tracking and reproducibility
- API keys and credentials should be kept secure and never committed to version control
- Sources without API keys will be gracefully skipped with warnings