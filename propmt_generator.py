from langchain_core.prompts import PromptTemplate


promt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If the context is insufficient, just say I don't know.

    {context}
    Question: {question}
""",
input_variables=["context", "question"],
validate_template= True

)

promt.save("./prompt_template.json")