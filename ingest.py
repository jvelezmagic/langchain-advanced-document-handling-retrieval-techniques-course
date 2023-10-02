from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import SQLRecordManager, index
from langchain.vectorstores import Chroma

from src.langchain_docs_loader import load_langchain_docs_for_app

load_dotenv()


def ingest():
    docs = load_langchain_docs_for_app()

    vectorstore = Chroma(
        collection_name="langchain_docs_app",
        persist_directory="data/chroma/langchain_docs_app/",
        embedding_function=OpenAIEmbeddings(),
    )

    record_manager = SQLRecordManager(
        db_url="sqlite:///data/langchain_docs_app.db",
        namespace="chroma/langchain_docs_app",
    )

    record_manager.create_schema()

    index(
        docs_source=docs,
        record_manager=record_manager,
        vector_store=vectorstore,
        cleanup="full",
        batch_size=1000,
    )


if __name__ == "__main__":
    ingest()
