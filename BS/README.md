# BS Data Analysis

A collection of NLP tools for analyzing chatbot intent data, including statistical analysis and semantic similarity measurements.

## Project Structure

```
BS/
├── src/                     # Source code
│   ├── __init__.py          # Makes src a proper Python package
│   ├── statistical_nlp_analysis.py  # Statistical NLP analysis
│   ├── semantic_intent_similarity.py  # Semantic intent similarity analysis
│   ├── utils.py             # Utility functions
│   ├── download_nltk_data.py  # Script to download required NLTK data
│   └── main.py              # Main entry point
├── BS_data/                 # Data files (CSV)
├── BS_dataframes/           # Pandas DataFrames (serialized)
├── utterances_tests/        # Utterances and test data
│   ├── utterances.json      # Training utterances
│   └── tests.json           # Test utterances
├── pie_plots/               # Generated visualizations
├── Dockerfile               # Docker container definition
├── Makefile                 # Build and run commands
└── requirements.txt         # Python dependencies
```

## Setup & Usage

### Using Docker (Recommended)

1. Build the Docker image:
   ```
   make build
   ```

2. Run a specific analysis:
   ```
   make nlp      # Run statistical NLP analysis
   make similarity  # Run semantic similarity analysis
   ```

3. Or get an interactive shell:
   ```
   make run
   ```

### Running Manually

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Download NLTK data:
   ```
   python -m src.download_nltk_data
   ```

3. Run analyses:
   ```
   python -m src.main nlp         # Statistical NLP analysis
   python -m src.main similarity  # Semantic similarity analysis
   python -m src.main all         # Run both analyses
   ```

## Features

### Statistical NLP Analysis

- Sentence/word length statistics
- Lexical richness metrics
- Outlier detection
- Train/Test set consistency metrics
- Visualization of distributions

### Semantic Intent Similarity

- Intent embedding centroids
- Intra-set similarity matrices
- Cross-set similarity measurement
- Visualization through heatmaps

## Data Processing

The tools process data from multiple sources:
- JSON files for intent utterances (train/test sets)
- CSV files in the BS_data directory

Results are saved as:
- JSON files with metrics and analysis results
- PNG visualizations (boxplots, heatmaps)
