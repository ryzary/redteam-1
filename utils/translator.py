from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="llama3", temperature=0)

def translate_sentence(sentence, languages):
    """Translate a sentence into multiple languages using an LLM."""
    prompt = f"Translate the following sentence into {', '.join(languages)}:\n\n'{sentence}'\n\nReturn JSON."
    response = llm.invoke(prompt)
    return response.content