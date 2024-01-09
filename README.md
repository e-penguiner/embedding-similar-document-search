# Document Search with Embeddings

Verified on Python 3.12.1.

## Quick start

At the first, install dependencies as following,

```shell
python -m venv .venv
```

Activate the virtual environment.

- Windows

```shell
.\.venv\Scripts\activate
pip install -r requirements.txt
```

- Linux
  
```shell
source .venv/bin/activate
pip install -r requirements.txt
```

Then prepare jsonl file to the similar document search.

```shell
python create_embeddings.py --excel-path=<excel_path> --header=<header_name> --sheet-name=<sheet_name> --model=<model> --api-key=<YOUR_OPENAI_API_KEY> --jsonl=<output_jsonl>
```

Search similar docs for your query.

```shell
python search_similar_doc.py --embedding-file=<josnl_path> --query=<your_query> --top=<num> --api-key=<YOUR_OPENAI_API_KEY>
```

Custom Hook

```shell
python search_similar_doc.py --embedding-file=<josnl_path> --query=<your_query> --top=<num> --hook=hook.yaml --api-key=<YOUR_OPENAI_API_KEY>
```
