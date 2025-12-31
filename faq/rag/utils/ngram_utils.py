import re
from typing import List, Set

def generate_ngrams(text: str) -> Set[str]:
    """
    Generate unigrams, bigrams, and trigrams from the given text.
    Standardizes whitespace, removes punctuation, and converts to lowercase.
    
    Format:
    - Unigrams: {word}
    - Bigrams: {word1,word2}
    - Trigrams: {word1,word2,word3}
    
    Args:
        text: The input string to process.
        
    Returns:
        A set of formatted n-gram strings.
    """
    if not text:
        return set()
    
    # Clean text: remove punctuation, lowercase, and tokenize by whitespace
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()
    
    ngrams = set()
    
    # Generate Unigrams: {word}
    for word in words:
        ngrams.add(f"{{{word}}}")
    
    # Generate Bigrams: {word1,word2}
    if len(words) >= 2:
        for i in range(len(words) - 1):
            ngrams.add(f"{{{words[i]},{words[i+1]}}}")
            
    # Generate Trigrams: {word1,word2,word3}
    if len(words) >= 3:
        for i in range(len(words) - 2):
            ngrams.add(f"{{{words[i]},{words[i+1]},{words[i+2]}}}")
            
    return ngrams

def get_ngram_overlap(set1: Set[str], set2: Set[str]) -> float:
    """
    Calculate the percentage overlap between two sets of n-grams.
    Specifically: (Intersection / count of set2) where set2 is usually the query n-grams.
    
    Args:
        set1: Set of n-grams (usually from the FAQ question).
        set2: Set of n-grams (usually from the user query).
        
    Returns:
        Overlap percentage as a float between 0.0 and 1.0.
    """
    if not set2:
        return 0.0
    
    intersection = set1.intersection(set2)
    return len(intersection) / len(set2)
