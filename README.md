# OpenSearch-RAG-Sample

OpenSearch と OpenAI を使用した RAG サンプル

## .env ファイル設定

```
# OpenSearchのADMINパスワード、暗号強度の高いパスワードでないと起動できない。
OPENSEARCH_INITIAL_ADMIN_PASSWORD=<admin password>

# OpenSearchインデックス名
OPENSEARCH_INDEX=legal_rag
# OpenSearchホスト名
OPENSEARCH_HOST=localhost
# OpenSearchポート番号
OPENSEARCH_PORT=9200

# OpenAI API Key
OPENAI_API_KEY=<API KEY>

# OpenAI GPTモデル
OPENAI_MODEL=gpt-4-vision-preview

# OpenAI Embeddingモデル
OPENAI_EMBDDING=text-embedding-3-small

# XMLドキュメントパス
XML_DOCUMENT_PATH=<xml path>

# HTMLドキュメントパス
HTML_DOCUMENT_PATH=<html path>

# Hugging Face Embeddingに使用するモデルのローカル保存先
MODEL_DIR=models
# Hugging Face Embeddingに使用するモデル名
MODEL_NAME=sentence-transformers/stsb-xlm-r-multilingual

# Embedding時のチャンクサイズとオーバーラップの指定
CHUNK_SIZE=2024
CHUNK_OVERLAP=256

# tokenizer の dead lock warning を回避
TOKENIZERS_PARALLELISM=false
```

## Embedding モデルをダウンロード

ローカルに Hugging Face Embedding モデルデータをダウンロード

```

$ python hf_model_downloader.py

```

## ドキュメント読み込み

法令データをダウンロードして展開
https://elaws.e-gov.go.jp/download/

```

$ python legal_doc_loader.with_hf.py

```

### Q&A

```
$ python question_hf.py

＞ 質問を入力してください (終了するには 'exit' と入力):
森林に関する法令をリストアップしてください
--- Answer ---
1. 森林法（昭和二十六年法律第二百四十九号）
2. 森林・林業基本法（昭和三十九年法律第百六十一号）
3. 森林・林業基本計画

これらの法令は森林や林業に関する施策や計画を定めています。

＞ 質問を入力してください (終了するには 'exit' と入力):
森林・林業基本法の概要を教えてください
--- Answer ---
森林・林業基本法は森林及び林業に関する施策についての基本理念及び責務を定め、森林及び林業に関する施策を総合的かつ計画的に
推進し、国民生活の安定向上及び国民経済の健全な発展を目的としています


質問を入力してください (終了するには 'exit' と入力):
exit
プログラムを終了します。
```
