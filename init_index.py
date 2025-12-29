import util
from opensearchpy import OpenSearch


def main():
    env = util.load_environ()

    client = opensearch_client(env)
    mapping = get_mapping()
    init_index(client, mapping, env)


def opensearch_client(env):

    host = env["OPENSEARCH_HOST"]
    port = env["OPENSEARCH_PORT"]
    auth = ("admin", env["OPENSEARCH_INITIAL_ADMIN_PASSWORD"])

    # OpenSearchクライアントのインスタンスを作成
    client = OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_compress=True,  # HTTP圧縮を有効にする
        http_auth=auth,
        use_ssl=True,  # HTTPSを使用する場合はTrueに設定
        verify_certs=False,  # SSL証明書を検証する場合はTrueに設定
        ssl_assert_hostname=False,  # SSLホスト名の検証を行わない場合はFalseに設定
        ssl_show_warn=False,  # SSL関連の警告を非表示にする場合はFalseに設定
    )
    return client


def get_mapping():
    return {
        "settings": {
            "index": {
                "analysis": {
                    "char_filter": {
                        "normalize": {
                            "type": "icu_normalizer",
                            "name": "nfkc",
                            "mode": "compose",
                        }
                    },
                    "tokenizer": {
                        "ja_kuromoji_tokenizer": {
                            "mode": "search",
                            "type": "kuromoji_tokenizer",
                        }
                    },
                    "analyzer": {
                        "kuromoji_analyzer": {
                            "tokenizer": "ja_kuromoji_tokenizer",
                            "filter": [
                                "kuromoji_baseform",
                                "kuromoji_part_of_speech",
                                "cjk_width",
                                "ja_stop",
                                "kuromoji_stemmer",
                                "lowercase",
                            ],
                        }
                    },
                }
            }
        },
        "mappings": {
            "properties": {
                "metadata": {
                    "properties": {
                        "source": {
                            "type": "text",
                            "fields": {"keyword": {"type": "keyword"}},
                        }
                    }
                },
                "text": {
                    "type": "text",
                    "analyzer": "kuromoji_analyzer",
                    "fields": {"keyword": {"type": "keyword"}},
                },
                "vector_field": {
                    "type": "knn_vector",
                    "dimension": 768,
                    "method": {
                        "engine": "nmslib",
                        "space_type": "l2",
                        "name": "hnsw",
                        "parameters": {"ef_construction": 512, "m": 16},
                    },
                },
            }
        },
    }


def init_index(client, mapping, env):

    index_name = env["OPENSEARCH_INDEX"]

    # インデックスが存在するか確認
    if client.indices.exists(index=index_name):
        # ユーザーにインデックスの初期化を確認
        user_input = input(
            f"インデックス '{index_name}' は既に存在します。初期化しますか？ (y/n): "
        )
        if user_input.lower() == "y":
            print(f"'{index_name}' インデックスを削除...")
            client.indices.delete(index=index_name)
            print(f"'{index_name}' インデックスを作成...")
            response = client.indices.create(index=index_name, body=mapping)
            print("Response:", response)
        else:
            print("インデックスの変更は無しで終了しました。")
    else:
        # インデックスが存在しない場合は新規作成
        print(f"'{index_name}' インデックスを作成...")
        response = client.indices.create(index=index_name, body=mapping)
        print("Response:", response)


if __name__ == "__main__":
    main()
