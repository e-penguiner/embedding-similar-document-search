import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import argparse

import embedding_utils


def create_embedding_pair(
    api_key,
    excel_path,
    header,
    sheet_name=None,
    model="text-embedding-ada-002",
):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df_embedding = pd.DataFrame()
    df_embedding["text"] = df[header].dropna()
    print(df_embedding)

    client = OpenAI(api_key=api_key)
    tqdm.pandas(desc="Generating embeddings")
    df_embedding["embedding"] = df_embedding["text"].progress_apply(
        lambda txt: embedding_utils.generate_embedding(client, txt, model)
    )

    return df_embedding


def main():
    parser = argparse.ArgumentParser(description="Generate Embeddings from Excel Data")
    parser.add_argument(
        "--excel-path", type=str, required=True, help="Path to the Excel file"
    )
    parser.add_argument(
        "--header", type=str, required=True, help="Header name for the text column"
    )
    parser.add_argument(
        "--sheet-name", type=str, default=None, help="Name of the Excel sheet"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="text-embedding-ada-002",
        help="Model to use for generating embeddings",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="Your OpenAI API key",
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default="embedding.jsonl",
        help="Path to the output JSONL file",
    )
    args = parser.parse_args()

    result_df = create_embedding_pair(
        api_key=args.api_key,
        excel_path=args.excel_path,
        header=args.header,
        sheet_name=args.sheet_name,
        model=args.model,
    )
    result_df.to_json(args.jsonl, orient="records", lines=True)


if __name__ == "__main__":
    main()
