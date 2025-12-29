import util
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA


def main():

    env = util.load_environ()

    vectorstore = connect_vectorstore(env)

    chain = retrieval_chain(env, vectorstore)

    qa(chain)


def connect_vectorstore(env):
    """ベクターストアと接続

    Args:
        env (dict): 環境変数

    Returns:
        OpenSearch: ベクターストア
    """
    model_path = f"{env['MODEL_DIR']}/{env['MODEL_NAME']}"
    hf = HuggingFaceEmbeddings(model_name=model_path)

    vectorstore = OpenSearchVectorSearch(
        opensearch_url=f'{env["OPENSEARCH_HOST"]}:{env["OPENSEARCH_PORT"]}',
        http_auth=("admin", env["OPENSEARCH_INITIAL_ADMIN_PASSWORD"]),
        index_name=env["OPENSEARCH_INDEX"],
        embedding_function=hf,
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
        bulk_size=10000,
        timeout=7200,
    )

    return vectorstore


def retrieval_chain(env, vectorstore):
    """LLM およびリトリーバーからチェーンをロードする

    Args:
        env (dict): 環境変数
        vectorstore: ベクターストア

    Returns:
        BaseConversationalRetrievalChain: チェーン
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    # llm = ChatOpenAI(model_name=env["OPENAI_MODEL"], temperature=0)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.8)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )

    return chain


def qa(chain):
    """QA実行

    Args:
        chain (BaseConversationalRetrievalChain): _description_
    """
    chat_history = []
    while True:
        query = input("\n\n＞ 質問を入力してください (終了するには 'exit' と入力): \n")
        if query.lower() == "exit":
            print("プログラムを終了します。")
            break
        print("--- Answer ---")
        response = chain.invoke({"query": query, "chain_history": chat_history})
        print(response["result"])
        chat_history = [(query, response["result"])]


if __name__ == "__main__":
    main()
