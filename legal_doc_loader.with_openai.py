import os
import time
import pathlib
import datetime
import splitter
import util
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredXMLLoader
from langchain_community.vectorstores import OpenSearchVectorSearch


def main():
    env = util.load_environ()

    document_load(env)


def document_load(env):
    """ドキュメントロード
    ベクターストアにXML文書を格納

    Args:
        env (dict): 環境変数
    """
    embeddings = OpenAIEmbeddings(
        model=env["OPENAI_EMBDDING"], openai_api_key=env["OPENAI_API_KEY"]
    )

    xml_doc_path = pathlib.Path(env["XML_DOCUMENT_PATH"])
    xml_file_paths = []

    for xml_path in xml_doc_path.glob("**/*.xml"):
        xml_file_paths.append(str(xml_path))

    # 読み込む件数の制限
    xml_file_paths = xml_file_paths[-50:]

    cnt = 0
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, "JST")
    now = datetime.datetime.now(JST)
    print(now)

    for xml_file in xml_file_paths:
        loader = UnstructuredXMLLoader(xml_file)
        docs = loader.load()

        text_splitter = splitter.recursive_character()
        split_docs = text_splitter.split_documents(docs)

        split_count = len(split_docs)
        cnt = cnt + 1
        print(str(cnt) + " : " + str(split_count) + " : " + xml_file)

        n = 0
        for split_doc in split_docs:
            db = OpenSearchVectorSearch.from_documents(
                [split_doc],
                embedding=embeddings,
                opensearch_url=f'{env["OPENSEARCH_HOST"]}:{env["OPENSEARCH_PORT"]}',
                http_auth=("admin", env["OPENSEARCH_INITIAL_ADMIN_PASSWORD"]),
                index_name=env["OPENSEARCH_INDEX"],
                use_ssl=True,
                verify_certs=False,
                ssl_show_warn=False,
                bulk_size=10000,
                timeout=7200,
            )
            db.client.indices.refresh(index=env["OPENSEARCH_INDEX"])
            # OpenAI API tokens per min(TPM) エラー対策のためのwait
            if n > 10000:
                time.sleep(1)
                n = 0
            else:
                n = n + 1

    now = datetime.datetime.now(JST)
    print(now)


if __name__ == "__main__":
    main()
