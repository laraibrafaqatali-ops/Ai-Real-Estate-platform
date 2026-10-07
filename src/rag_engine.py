import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class PropertyRAGEngine:
    def __init__(self, docs_dir='data/docs'):
        self.docs_dir = docs_dir
        self.rag_chain = None
        self._initialize()

    def _initialize(self):
        if not os.path.exists(self.docs_dir):
            os.makedirs(self.docs_dir, exist_ok=True)
            with open(f'{self.docs_dir}/buying_guide.md', 'w') as f:
                f.write("# Property Buying Guide\nAlways verify ownership documents, check municipal tax records, evaluate neighborhood infrastructure, and perform structural inspections before making an offer.")

        loader = DirectoryLoader(self.docs_dir, glob="*.md", loader_cls=TextLoader)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)

        embeddings = FastEmbedEmbeddings()
        vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        llm = ChatGroq(
            model_name="openai/gpt-oss-20b",
            api_key=os.getenv("GROQ_API_KEY")
        )

        template = """You are an AI assistant specialized in real estate documentation. Use the retrieved context to answer user questions. If the answer is not present in context, state that clearly.

Context:
{context}

Question:
{question}

Answer:"""

        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(documents):
            return "\n\n".join(doc.page_content for doc in documents)

        self.rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def query_docs(self, question: str) -> str:
        return self.rag_chain.invoke(question)