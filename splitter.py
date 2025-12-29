from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

_JAPANESE_SEPARATORS = [
    "\n\n",
    "\n",
    "。",
    "、",
    "「",
    "」",
    "！",
    "？",
    "『",
    "』",
    "（",
    "）",
    " ",
    "",
]


def recursive_character(env):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=env["CHUNK_SIZE"],
        chunk_overlap=env["CHUNK_OVERLAP"],
        separators=_JAPANESE_SEPARATORS,
    )
    return text_splitter


def huggingface_tokenizer(env):
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        encoding_name="cl100k_base",
        chunk_size=env["CHUNK_SIZE"],
        chunk_overlap=env["CHUNK_OVERLAP"],
        separators=_JAPANESE_SEPARATORS,
    )
    return text_splitter


def token_text(env):
    text_splitter = TokenTextSplitter(
        chunk_size=env["CHUNK_SIZE"],
        chunk_overlap=env["CHUNK_OVERLAP"],
    )
    return text_splitter


def stf_token(model_path, env):
    text_splitter = SentenceTransformersTokenTextSplitter(
        chunk_size=env["CHUNK_SIZE"],
        chunk_overlap=env["CHUNK_OVERLAP"],
        model_name=model_path,
    )
    return text_splitter
