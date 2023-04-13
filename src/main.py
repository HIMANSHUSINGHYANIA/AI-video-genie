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
        self.collection = self.create_collection(client=self.client, collection=collection)

    @classmethod
    def create_collection(cls, client: chromadb.Client, embedding_function: EmbeddingFunction = None,
                          collection: str = None):
        if collection:
            return client.get_collection(name=f"{collection}",
                                         embedding_function=embedding_function or openai_ef)
        else:
            return client.create_collection(name="video_library",
                                            embedding_function=embedding_function or openai_ef)

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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        return texts

    @staticmethod
    def add_video(video_url: str):
        video = Video(video_url)
        return video

    @staticmethod
    def chunk_transcribe_video(video_url: str) -> Tuple[List[Dict[str, str]], StrPath]:
        video = Video(video_url).download()
        chunks = video.chunks()
        transcriptions = video.transcribe(chunks)
        merged_transcription = video.merge_transcriptions(transcriptions)
        return (transcriptions, merged_transcription)

    @staticmethod
    def genius(text_path: StrPath, question: str):
        loader = TextLoader(text_path)
        documents = loader.load()
        texts = Genius.text_split(documents)
        embeddings = OpenAIEmbeddings()
        vectordb = embeddings(texts)
        answer = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=vectordb)
        return answer.run(question)


if __name__ == "__main__":
    genius = Genius(collection="video_library")
    # genius.create_collection()

    print(genius.collection.get(ids=[str(i) for i in range(9999)],
        include=["documents"]))

    chunks_merged = genius.index_chunks("https://www.youtube.com/watch?v=ZZIR1NGwy-s&ab_channel=BeyondFireship")
    query = "How to make my pagespeed perfect?"
    res = genius.query(n_results=2, query_texts=[query])
    print(res)
    print("----*"*90)
    print("Your genius is ready to serve you!\n")
    print(f"You should watch from this minute: {int(res['metadatas'][0][0]['sec'])/60} and this minute: {int(res['metadatas'][0][1]['sec'])/60}")
    print(genius.genius(chunks_merged, query))


    """Not relevant now"""

    # vectors = genius.query(n_results=2, query_texts=["Code of conduct"])
    # res = {"document": vectors["documents"],
    #        "mins": vectors["metadatas"]}
    # res = {'ids': [['9520992337', '7902261223']], 'embeddings': None, 'documents': [[
    #                                                                                     "And so you think weed is legal in this country? I guess so, yeah. You guess so? Is it not? I don't know. Well, you're going to have to watch this video to find out. Stay tuned. Why are there so many weed and cannabis shops in Prague, Czech Republic? Well, the very short answer is because there's a lot of tourists here. What do you guys think about the cannabis shops here? Have you seen a few? We literally just landed. You just landed. 20 minutes ago, first time here. I don't know what the law is or anything like that. You don't know what the law is? We weren't anticipating there were anything like this. Oh, you weren't anticipating. Okay, okay. We're more curious to what is illegal in this country. Well, does it seem that it would be illegal or not? It certainly does. It certainly does. It gives the Amsterdam vibes. It gives the Amsterdam vibes. Come on in and you know. Now, the question that obviously pops in your head is, is weed legal in this country? Well, the answer to it is not just yes and no. It's a bit more complicated than that and we'll get to it throughout the video. But let's first look at how these stores actually target towards their customers. Usually, they target.",
    #                                                                                     "You're buying? No? Get out. Remember this store? We already mentioned it. It has a one-star review on Google. Now I know why. Now the following two stores caught our eye for different reasons. One had the Comic Sans written sign that said marijuana and the other one seemed quite welcoming and had the sign THC on it, which is quite important for this part. So I walked in and I asked if they sell wheat that contains THC. This is what they said. You have THC wheat? 20%? So it's legal here? It's not? Oh, nobody cares? So how can you sell it? You have the license? You have license? Oh. So, but can I smoke it on the street? Nobody cares. Is this as good? Wow, it smells good. It's THC? THC."]],
    #        'metadatas': [[{'min': 0}, {'min': 4}]], 'distances': [[0.315494567155838, 0.32840782403945923]]}
    #
    # for k, v in res.items():
    #     position = ["document"].find(query)
    #     if position != -1:
    #         print(f'The substring appears at position {position} {len(i.split())} in the string.')
    #     else:
    #         print(f'does not appear')
