#!/usr/bin/env python3
"""
List all documents in Python
"""

def list_all(mongo_collection):
    """
    Return all documents in the given MongoDB collection.
    If no documents found, returns an empty list.
    Args:
        mongo_collection: pymongo collection object

    Returns:
        List of documents (each document is a dictionary)
    """
    documents = list(mongo_collection.find())
    return documents
