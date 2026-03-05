import os
from langchain_huggingface import HuggingFaceEmbeddings,ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# 1️⃣ CONFIGURATION

DB_FAISS_PATH = "vectorstore/db/faiss"
token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# 2️⃣ LOAD EMBEDDING MODEL

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 3️⃣ LOAD FAISS DATABASE

try:
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
except Exception as e:
    print(f"Error loading FAISS DB: {e}")
    exit()

# 4️⃣ CREATE RETRIEVER

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5, "lambda_mult": 0.25}
)

# 5️⃣ STRICT CUSTOM PROMPT

custom_prompt = """
You are a retrieval-based assistant.

Context:
{context}

Question:
{question}

STRICT RULES:
- Use ONLY the provided context.
- Do NOT add external knowledge.
- Do NOT fabricate details.
- Do NOT assume missing data.
- If answer is not found in context, reply exactly:
Sorry, I don't have any knowledge regarding this.

Return only the final answer.
"""

rewrite_prompt = """
Rewrite the user question to make it clearer and easier for a medical document search system.Don't user words like here are the queries etc just rewrite the question

User question:
{question}

Improved question:
"""
improved_query = PromptTemplate(
    template= rewrite_prompt,
    input_variables= ['question']
) 

prompt = PromptTemplate(
    template=custom_prompt,
    input_variables=["context", "question"]
)

# 6️⃣ LOAD LLM

llm = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.2-3B-Instruct",
    task= "text-generation",
    huggingfacehub_api_token=token,
    temperature=0.2,
    max_new_tokens=512,
)
model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()
# 7️⃣ FORMAT DOCUMENTS


def format_docs(docs):
    if not docs:
        return ""

    formatted_docs =  "\n\n".join(doc.page_content for doc in docs)
    return formatted_docs

rewrite_chain = improved_query | model | parser

def improve_query(user_query):
    return rewrite_chain.invoke({"question": user_query})

# 8️⃣ BUILD MODERN LCEL RAG CHAIN
rag_chain = (
    {
        "context": RunnableLambda(lambda x: x["question"])
                   | retriever
                   | RunnableLambda(format_docs),

        "question": RunnableLambda(lambda x: x["question"])
    }
    | prompt
    | model
    | parser
)

# 9️⃣ RUN
def get_answer(user_query):
    try:
        # rewrite user query
        improved = improve_query(user_query)

        print("Original Query:", user_query)
        print("Improved Query:", improved)

        # run RAG
        response = rag_chain.invoke({
            "question": improved
        })
        return response

    except Exception as e:
        return f"Error: {str(e)}"