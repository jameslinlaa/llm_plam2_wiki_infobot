import os

from langchain.document_loaders import GitLoader
from git import Repo
from langchain.embeddings import OpenAIEmbeddings, VertexAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import WikipediaLoader
import pinecone


pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)


def ingest_docs() -> None:
    docs = WikipediaLoader(query="Hunter X Hunter", load_max_docs=2, lang="zh-tw").load()

    print(f"loaded {len(docs)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
    )

    documents = text_splitter.split_documents(docs)

    print(f"Going to add {len(documents)} to Pinecone")

    embeddings = VertexAIEmbeddings()

    chunk_size = 5
    for i in range(0, len(documents), chunk_size):
        print(f"iteration {i}/{len(documents)/chunk_size}...")
        chunked_documents = documents[i : i + chunk_size]
        Pinecone.from_documents(
            chunked_documents, embeddings, index_name=os.environ["PINECONE_INDEX_NAME"]
        )
    print("****Loading to vectorestore done ***")


if __name__ == "__main__":
    ingest_docs()
