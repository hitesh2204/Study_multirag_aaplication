# rag_pipeline/rag_chain.py
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser

def get_rag_chain(retriever, llm):
    # 1. Prompt template
    prompt = PromptTemplate.from_template(
        """You are a helpful study assistant. Use the following context to answer the question:
        
        Context:
        {context}
        
        Question:
        {question}
        
        Helpful Answer:"""
    )

    # Wrap retriever in a function that returns string context
    def retrieve_and_format(input_dict):
        docs = retriever.get_relevant_documents(input_dict["question"])
        context = "\n\n".join(doc.page_content for doc in docs)
        return {"context": context, "question": input_dict["question"]}
    
    # 2. Define chain
    chain = (
        {"context": retriever, "question": lambda x: x["question"]}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
