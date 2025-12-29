import time
import pathlib
import datetime
import torch
import splitter
import util
from langchain_community.document_loaders import UnstructuredXMLLoader
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_community.embeddings import HuggingFaceEmbeddings


def main():
    env = util.load_environ()

    document_load(env)


def document_load(env):
    """ドキュメントロード
    ベクターストアにXML文書を格納

    Args:
        env (dict): 環境変数
    """
    # Embeddingモデルのローカルパス
    model_path = f"{env['MODEL_DIR']}/{env['MODEL_NAME']}"

    # Embeddingモデルの計算を実行する機器
    embedding_device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(f"USE DEVICE: {embedding_device}")
    hf = HuggingFaceEmbeddings(
        model_name=model_path, model_kwargs={"device": embedding_device}
    )

    xml_doc_path = pathlib.Path(env["XML_DOCUMENT_PATH"])
    xml_file_paths = []

    for xml_path in xml_doc_path.glob("**/*.xml"):
        xml_file_paths.append(str(xml_path))

    # テストで読み込む件数の制限する場合
    # xml_file_paths = xml_file_paths[-100:]

    cnt = 0
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, "JST")
    now = datetime.datetime.now(JST)
    print(now)

    idx_cnt = 0
    for xml_file in xml_file_paths:
        loader = UnstructuredXMLLoader(xml_file)
        docs = loader.load()

        text_splitter = splitter.recursive_character(env)
        chunk = text_splitter.split_documents(docs)

        split_count = len(chunk)
        cnt = cnt + 1
        print(str(cnt) + " : " + str(split_count) + " : " + xml_file)

        idx_cnt = idx_cnt + split_count

        OpenSearchVectorSearch.from_documents(
            chunk,
            embedding=hf,
            opensearch_url=f'{env["OPENSEARCH_HOST"]}:{env["OPENSEARCH_PORT"]}',
            http_auth=("admin", env["OPENSEARCH_INITIAL_ADMIN_PASSWORD"]),
            index_name=env["OPENSEARCH_INDEX"],
            use_ssl=True,
            verify_certs=False,
            ssl_show_warn=False,
            engine="faiss",  # Faissインデックス
            ef_construction=512,  # 探索効率を決定するパラメータ
            m=40,  # HNSWアルゴリズムにおいて、各ノードが保持するエッジ（つながり）の最大数
            bulk_size=100000,
            timeout=7200,
        )
        # db.client.indices.refresh(index=env["OPENSEARCH_INDEX"])

        if idx_cnt > 4000:
            print("--- wait ---")
            time.sleep(30)
            idx_cnt = 0

    now = datetime.datetime.now(JST)
    print(now)


if __name__ == "__main__":
    main()
