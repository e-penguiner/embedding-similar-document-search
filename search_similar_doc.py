import pandas as pd
from openai import OpenAI
import numpy as np
import argparse

import embedding_utils


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search_docs(api_key, df, user_query, top_n=5):
    client = OpenAI(api_key=api_key)

    embedding = embedding_utils.generate_embedding(
        client,
        user_query,
        model="text-embedding-ada-002",
    )
    similarities = df["embedding"].apply(lambda x: cosine_similarity(x, embedding))
    top_indices = similarities.nlargest(top_n).index
    return [(df["text"][i], similarities[i]) for i in top_indices]


def main():
    parser = argparse.ArgumentParser(description="Document Search with Embeddings")
    parser.add_argument(
        "--embedding-file",
        type=str,
        required=True,
        help="The JSONL file with embeddings",
    )
    parser.add_argument(
        "--query", type=str, required=True, help="User query for searching documents"
    )
    parser.add_argument(
        "--top", type=int, default=5, help="Number of top documents to retrieve"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="Your OpenAI API key",
    )

    args = parser.parse_args()
    df = pd.read_json(args.embedding_file, orient="records", lines=True)
    print(
        search_docs(api_key=args.api_key, df=df, user_query=args.query, top_n=args.top)
    )


if __name__ == "__main__":
    main()
