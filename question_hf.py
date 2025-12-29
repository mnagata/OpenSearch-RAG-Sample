import torch
import util
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


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
        search_type="similarity", search_kwargs={"k": 10, "score_threshold": 0.8}
    )

    llm = ChatOpenAI(model_name=env["OPENAI_MODEL"], temperature=0)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_template()},
        condense_question_prompt=condense_qa_prompt(),
        condense_question_llm=llm,
    )

    return chain


def condense_qa_prompt():
    """生成質問のプロンプト
    LLMが質問と会話履歴を受け取って、質問の言い換え（生成質問）を行う
    """
    condense_qa_template = """
次の会話とフォローアップの質問を考慮して、フォローアップの質問を元の言語で独立した質問に言い換えます。チャット履歴がない
場合は、質問を独立した質問に言い換えてください。

チャットの履歴:
{chat_history}

フォローアップの質問:
{question}

言い換えられた独立した質問:"""

    return PromptTemplate.from_template(condense_qa_template)


def qa_template():
    """回答プロンプト
    質問と関連情報を合わせてLLMに投げる
    """
    prompt_template = """
あなたは質問応答タスクのアシスタントです。 取得したコンテキストの次の部分を使用して質問に答えます。 答えがわからない場合
は、わからないと言ってください。 
コンテキスト：
{context}

質問: 
{question}:"""

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    return prompt


def qa(chain):
    """QA実行

    Args:
        chain (BaseConversationalRetrievalChain): _description_
    """
    while True:
        query = input("\n\n＞ 質問を入力してください (終了するには 'exit' と入力): \n")
        if query.lower() == "exit":
            print("プログラムを終了します。")
            break
        print("--- Answer ---")
        response = chain.invoke({"question": query})
        print(response["answer"])
        # 参照ソースを表示
        # docs = response["source_documents"]
        # urls = []
        # for doc in docs:
        #    urls.append(doc.metadata["full_text_url"])
        # print("参照URL")
        # source_urls = list(urls)
        # for url in source_urls:
        #    print(url)


if __name__ == "__main__":
    main()
