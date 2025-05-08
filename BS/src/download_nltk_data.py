#!/usr/bin/env python3
"""
download_nltk_data.py

Downloads necessary NLTK data packages for the NLP analysis.
"""
import nltk
import sys


def download_nltk_data():
    """Download required NLTK data packages."""
    print("Downloading NLTK data packages...")
    nltk.download('punkt')
    print('NLTK punkt (all languages) downloaded')


if __name__ == "__main__":
    download_nltk_data()
    sys.exit(0)
