import os

import chromadb
from chromadb.config import Settings
from chromadb.api.types import EmbeddingFunction
from chromadb.utils import embedding_functions
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from typing import List, Dict, Tuple
from pydantic.typing import StrPath
from langchain.chains import VectorDBQA
from langchain.llms import OpenAI
from src.utils import Video

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model_name="text-embedding-ada-002"
)


class Genius:
    """
    Chroma API: https://docs.trychroma.com/usage-guide
    """

    def __init__(self, collection: str = None):
        self.client = chromadb.Client(Settings(
            persist_directory="chroma_db",
            chroma_db_impl="duckdb+parquet",
            anonymized_telemetry=False
        ))
        self.collection = self._create_collection(client=self.client, name=collection)

    @classmethod
    def _create_collection(cls, client: chromadb.Client, embedding_function: EmbeddingFunction = openai_ef, name: str = None):
        try:
            return client.get_collection(name=name or "video_libary",
                                         embedding_function=embedding_function)
        except Exception as e:
            print(e)
            return client.create_collection(name=name or "video_libary",
                                            embedding_function=embedding_function)

    def add_document(self, docments: List[str], metadatas: List[Dict[str, str]], ids: List[str],
                     embeddings: List[List[float]] = None):
        if not embeddings:
            self.collection.add(
                documents=docments,
                metadatas=metadatas,
                ids=ids
            )
        else:
            self.collection.add(
                documents=docments,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

    def query(self, n_results: int = 10, where: str = None, where_document: str = None,
              query_embedding: List[List[float]] = None, query_texts: List[str] = None):

        if not query_embedding and not query_texts:
            raise ValueError("Either query_embedding or query_texts must be provided")

        if query_embedding:
            return self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                where=where or None,
                where_document=where_document or None
            )
        else:
            return self.collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where=where or None,
                where_document=where_document or None
            )

    def get_by_id(self, ids: List[str] = None, where: Dict[str, str] = None, where_document: Dict[str, str] = None):

        if not ids and not where and not where_document:
            raise ValueError("Either ids or where or where_document must be provided")

        return self.collection.get(
            ids=["id1", "id2", "id3", ...],
            where={"style": "style1"}
        )

    def index_chunks(self, video_url: str):
        data = Genius.chunk_transcribe_video(video_url)
        for chunk in data[0]:
            self.add_document(docments=chunk["transcript"],
                              ids=chunk["id"], metadatas=[chunk["min"]])  # , metadatas=list(chunk),
        return data[1]

    @staticmethod
    def text_split(documents: TextLoader):
        """
        Splits merged transcription into
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        return texts

    @staticmethod
    def add_video(video_url: str):
        video = Video(video_url)
        return video

    @staticmethod
    def chunk_transcribe_video(video_url: str) -> Tuple[List[Dict[str, str]], StrPath]:
        """
        Chunks video and transcribes it, returns the transcriped chunks and the merged transcription
        """
        video = Video(video_url).download()
        chunks = video.chunks()
        transcriptions = video.transcribe(chunks)
        merged_transcription = video.merge_transcriptions(transcriptions)
        return (transcriptions, merged_transcription)

    @staticmethod
    def genius(text_path: StrPath, question: str):
        """
        Deals with the question answering part upon the whole video transcript
        """
        loader = TextLoader(text_path)
        documents = loader.load()
        texts = Genius.text_split(documents)
        embeddings = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(texts, embeddings)
        answer = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=vectordb)
        return answer.run(question)


if __name__ == "__main__":
    genius = Genius(collection="video_library")

    chunks_merged = genius.index_chunks("https://www.youtube.com/watch?v=1BXO2FGcMjc&ab_channel=OtherPeople%27sComputer")

    query_video_chunks = "requirements and path to get SRE"
    query_merged_transcript = "Whats the prerequisites to get SRE? What is the described path to get SRE?"
    res = genius.query(n_results=10, query_texts=[query_video_chunks])
    print(res)
    print("----*"*90)
    print("Your genius is ready to serve you!\n")
    print(f"You should watch from this minute:")
    for i in range(len(res['metadatas'][0])):
        print(f"{int(res['metadatas'][0][i]['sec'])/60})")
    print("----*" * 90)
    print(genius.genius(chunks_merged, query_merged_transcript))
