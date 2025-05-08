#!/usr/bin/env python3
"""
main.py

Main entry point for BS data analysis, providing a command-line interface
to run the statistical NLP analysis and semantic intent similarity analysis.
"""
import argparse
from pathlib import Path
import sys
from src.statistical_nlp_analysis import main as nlp_analysis_main
from src.semantic_intent_similarity import main as semantic_similarity_main


def main():
    """Parse command line arguments and run the appropriate analysis."""
    parser = argparse.ArgumentParser(description="BS Data Analysis Tools")
    
    subparsers = parser.add_subparsers(dest='command', help='Analysis command to run')
    
    # NLP Analysis command
    nlp_parser = subparsers.add_parser('nlp', help='Run statistical NLP analysis')
    
    # Semantic similarity command
    similarity_parser = subparsers.add_parser('similarity', help='Run semantic intent similarity analysis')
    
    # Combined analysis command 
    combined_parser = subparsers.add_parser('all', help='Run both analyses')
    
    args = parser.parse_args()
    
    if args.command == 'nlp':
        print("Running statistical NLP analysis...")
        nlp_analysis_main()
    elif args.command == 'similarity':
        print("Running semantic intent similarity analysis...")
        semantic_similarity_main()
    elif args.command == 'all':
        print("Running all analyses...")
        print("\n=== Statistical NLP Analysis ===")
        nlp_analysis_main()
        print("\n=== Semantic Intent Similarity Analysis ===")
        semantic_similarity_main()
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
