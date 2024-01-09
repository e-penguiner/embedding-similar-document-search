import pandas as pd
from openai import OpenAI
import numpy as np
import argparse
import yaml

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


def prehook(api_key, config, query):
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "system", "content": config["prompt"]},
            {"role": "user", "content": query},
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response.choices[0].message.content


def hook(api_key, hook_file, query):
    with open(hook_file, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
        if "prehook" in config:
            return prehook(api_key=api_key, config=config["prehook"], query=query)
    return query


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
    parser.add_argument(
        "--hook", type=str, help="The YAML configuration file for the hook"
    )

    args = parser.parse_args()
    query = args.query
    if args.hook:
        query = hook(api_key=args.api_key, hook_file=args.hook, query=query)
    print("Converted query is [", query, "]")
    df = pd.read_json(args.embedding_file, orient="records", lines=True)
    print(search_docs(api_key=args.api_key, df=df, user_query=query, top_n=args.top))


if __name__ == "__main__":
    main()
