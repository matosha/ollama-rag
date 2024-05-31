import argparse,os,shutil
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_PATH = "./chroma"
DATA_PATH = "./data"

PROMPT_TEMPLATE = """
Answer the question by based only on the following context:

{context}

---

Answer the question based on the above context by following the below rules.
Rule 1: Create three parts with the following content: 
A "Description" part as short description of the complete answer.
A "Tooling" part which makes reference to Tenable and how it can be used to address the question and the corresponding NIST CSF Function. 
A "Solution" part as recommended code in either python, bash or command line which addresses the question. 
A "References" part At the end of the answer, add three references and page numbers which support the answer. Each reference has a "Source" description and "Page" as the location.

Rule 3: The entire answer must be a json object.

Question: {question}
Answer:
"""
class db_management:
    def __init__(self):
        self.CHROMA_PATH = "chroma"
        self.DATA_PATH = "data"
        self.db=None
        self.embeddings=None
    
    def load_documents(self,):
        document_loader = PyPDFDirectoryLoader(self.DATA_PATH)
        return document_loader.load()

    def split_documents(self,documents: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=6000,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_documents(documents)
    
    def embeddings_function(self,):
        if self.embeddings!=None:
            return self.embeddings
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'gpu'}
        encode_kwargs = {'normalize_embeddings': False}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        return self.embeddings

    def add_to_chroma(self,chunks: list[Document]):
        # Load the existing database.
        
        self.db = Chroma(
            persist_directory=self.CHROMA_PATH, embedding_function=self.embeddings
        )

        # Calculate Page IDs.
        chunks_with_ids = self.calculate_chunk_ids(chunks)

        # Add or Update the documents.
        existing_items = self.db.get(include=[])  # IDs are always included by default
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing documents in DB: {len(existing_ids)}")

        # Only add documents that don't exist in the DB.
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            print(f"👉 Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            self.db.add_documents(new_chunks, ids=new_chunk_ids)
            self.db.persist()
        else:
            print("✅ No new documents to add")


    def calculate_chunk_ids(self,chunks):

        # This will create IDs like "data/monopoly.pdf:6:2"
        # Page Source : Page Number : Chunk Index

        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"

            # If the page ID is the same as the last one, increment the index.
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            # Calculate the chunk ID.
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

            # Add it to the page meta-data.
            chunk.metadata["id"] = chunk_id

        return chunks

    def clear_database(self,):
        if os.path.exists(self.CHROMA_PATH):
            shutil.rmtree(self.CHROMA_PATH)



def query_rag(query_text: str):
    # Prepare the DB.
    db = Chroma(persist_directory=db_handler.CHROMA_PATH, embedding_function=db_handler.embeddings_function())

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text

db_handler= db_management()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_text", type=str, help="The query text.")
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--loaddb", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    
    if args.reset:
        db_handler.clear_database()
    
    if args.loaddb:
        documents = db_handler.load_documents()
        chunks = db_handler.split_documents(documents)
        db_handler.add_to_chroma(chunks)
    
    if args.query_text:
        query_text = args.query_text
        query_rag(query_text)
        while True:
            query_rag(input("Another question?:"))
    print("Closing. ")
