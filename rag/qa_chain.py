from langchain_community.llms import Ollama
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate

PROMPT_TEMPLATE = """You are a clinical decision support assistant. Use ONLY the provided context to answer the question.
If the context does not contain enough information, say "I cannot find sufficient evidence in the provided documents."
Always cite the source document names at the end of your answer.

Context:
{summaries}

Question: {question}

Answer (with citations):"""


def build_qa_chain(retriever):
    llm = Ollama(model="mistral", temperature=0.1)

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["summaries", "question"],
    )

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return chain
