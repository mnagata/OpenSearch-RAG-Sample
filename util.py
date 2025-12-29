import os
from dotenv import load_dotenv
from opensearchpy import OpenSearch


def load_environ():
    """環境変数ロード

    Returns:
        dict[str, str]: 環境変数dict
    """
    load_dotenv()

    env = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
        "OPENAI_MODEL": os.environ.get("OPENAI_MODEL", "gpt-4-vision-preview"),
        "OPENAI_EMBDDING": os.environ.get("OPENAI_EMBDDING", "text-embedding-3-small"),
        "MODEL_DIR": os.environ.get("MODEL_DIR", "models"),
        "MODEL_NAME": os.environ.get("MODEL_NAME", "stsb-xlm-r-multilingual"),
        "OPENSEARCH_INITIAL_ADMIN_PASSWORD": os.environ.get(
            "OPENSEARCH_INITIAL_ADMIN_PASSWORD", ""
        ),
        "OPENSEARCH_HOST": os.environ.get("OPENSEARCH_HOST", "localhost"),
        "OPENSEARCH_PORT": os.environ.get("OPENSEARCH_PORT", "9200"),
        "OPENSEARCH_INDEX": os.environ.get("OPENSEARCH_INDEX", ""),
        "XML_DOCUMENT_PATH": os.environ.get("XML_DOCUMENT_PATH", "xml"),
        "HTML_DOCUMENT_PATH": os.environ.get("HTML_DOCUMENT_PATH", "html"),
        "CHUNK_SIZE": int(os.environ.get("CHUNK_SIZE", "2048")),
        "CHUNK_OVERLAP": int(os.environ.get("CHUNK_OVERLAP", "256")),
    }
    return env


def check_index_exists(env):
    """
    インデックスの存在確認

    :param host: ホスト名
    :param index_name: インデックス名
    :return: 存在するならTrue
    """
    client = OpenSearch(
        hosts=[{"host": env["OPENSEARCH_HOST"], "port": env["OPENSEARCH_PORT"]}],
        http_auth=("admin", env["OPENSEARCH_INITIAL_ADMIN_PASSWORD"]),
        http_compress=True,
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
    )

    return client.indices.exists(index=env["OPENSEARCH_INDEX"])
