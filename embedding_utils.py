import re


def generate_embedding(client, text, model="text-embedding-ada-002"):
    response = client.embeddings.create(input=normalize_text(text), model=model)
    return response.data[0].embedding


def normalize_text(s, sep_token=" \n "):
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r". ,", "", s)
    s = s.replace("..", ".")
    s = s.replace(". .", ".")
    s = s.replace("\n", "")
    return s.strip()
