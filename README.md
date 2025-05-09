# Chatbot Data Analysis Toolkit

A comprehensive toolkit for analyzing chatbot data from Kore.ai, providing extended metrics, lexical analysis, and semantic insights for intents and entities.

## Overview

This repository contains scripts and tools for analyzing chatbot data, with a focus on:

1. **Extended Metrics Analysis**: Calculate detailed metrics for intents and entities from Kore.ai batch testing results
2. **Utterance Analysis**: Analyze utterances and test suite data through lexical, distribution, and semantic perspectives

## Project Structure

```
/
├── src/                     # Source code
│   ├── process_batch_raw.py      # Process raw batch testing files
│   ├── calculate_intent_metrics.py # Calculate intent metrics
│   ├── calculate_entity_metrics.py # Calculate entity metrics
│   ├── analyze_lexical.py        # Lexical analysis
│   ├── analyze_distribution.py   # Distribution analysis
│   ├── analyze_semantic.py       # Semantic analysis
│   └── utils.py                  # Utility functions
├── batch_testing_results/    # Batch testing results from Kore.ai
│   ├── raw/                  # Raw batch testing files
│   └── processed/            # Processed batch testing files
├── Results/                  # Generated results
│   ├── Intents/              # Intent metrics results
│   └── Entities/             # Entity metrics results
├── utterances/              # Utterance data in CSV format (one file per intent)
├── test_suites/             # Test suite data in CSV format (one file per intent)
└── dependencies.txt         # Python dependencies
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r dependencies.txt
   ```

## Usage

### 1. Extended Metrics Analysis from Kore.ai Batch Testing

This workflow provides detailed metrics for intents and entities from Kore.ai batch testing results.

#### Step 1: Process Raw Batch Testing Files

1. Upload your batch testing CSV files from Kore.ai to the `/batch_testing_results/raw/` directory.

2. Process the raw files to prepare them for analysis:
   ```bash
   python src/process_batch_raw.py
   ```
   This will generate processed files in the `/batch_testing_results/processed/` directory.

#### Step 2: Calculate Intent Metrics

Generate detailed intent metrics from the processed batch testing files:
```bash
python src/calculate_intent_metrics.py
```

This will generate CSV files in the `/Results/Intents/` directory with metrics such as:
- Precision, recall, and F1 score for each intent
- Support (number of test cases) for each intent
- Accuracy and other performance metrics
- Special handling for the "none" intent

#### Step 3: Calculate Entity Metrics

Generate detailed entity metrics from the processed batch testing files:
```bash
python src/calculate_entity_metrics.py
```

This will generate multiple CSV files in the `/Results/Entities/` directory:
- `*_entity_metrics.csv`: Accuracy and support for each entity type
- `*_entity_engine_summary.csv`: Distribution of engines (ML Engine vs FM Engine) for each entity type
- `*_entity_methods.csv`: Specific methods used for entity identification
- `*_entity_identification.csv`: Detailed breakdown of entity identification methods
- `*_entity_success_koreai.csv`: Utterance-level entity success metrics

### 2. Utterance and Test Suite Analysis

This workflow provides insights into the utterances and test suite data used in your chatbot.

#### Step 1: Prepare Utterance and Test Suite Data

1. Create CSV files for each intent and place them in the appropriate directories:
   - Training utterances: `/utterances/` (one CSV file per intent)
   - Test suite data: `/test_suites/` (one CSV file per intent)

#### Step 2: Lexical Analysis

Generate lexical information about your utterances:
```bash
python src/analyze_lexical.py
```

#### Step 3: Distribution Analysis

Analyze the distribution of utterances across intents:
```bash
python src/analyze_distribution.py
```

#### Step 4: Semantic Analysis

Perform semantic analysis on your utterances:
```bash
python src/analyze_semantic.py
```

## Output Files

### Intent Metrics

The intent metrics files (in `/Results/Intents/`) provide detailed performance metrics for each intent, including:
- Precision: How many of the predicted intents are correct
- Recall: How many of the actual intents are correctly identified
- F1 Score: Harmonic mean of precision and recall
- Support: Number of test cases for each intent
- Accuracy: Overall accuracy of intent recognition

### Entity Metrics

The entity metrics files (in `/Results/Entities/`) provide insights into entity recognition performance:

1. **Entity Metrics**: Accuracy and support for each entity type

2. **Entity Engine Summary**: Shows which engine (ML Engine or FM Engine) is used for each entity type

3. **Entity Methods**: Shows the specific methods used for entity identification within each engine

4. **Entity Identification**: Detailed breakdown of entity identification methods with hierarchical relationship

5. **Entity Success KoreAI**: Utterance-level entity success metrics

## Troubleshooting

### Common Issues

1. **Missing Directories**: If any of the required directories are missing, create them manually:
   ```bash
   mkdir -p batch_testing_results/raw batch_testing_results/processed Results/Intents Results/Entities utterances test_suites
   ```

2. **File Format Issues**: Ensure that your Kore.ai batch testing files are in the correct CSV format. If you encounter errors, check the column names and data format.

3. **Dependency Issues**: If you encounter errors related to missing packages, ensure all dependencies are installed:
   ```bash
   pip install -r dependencies.txt
   ```
