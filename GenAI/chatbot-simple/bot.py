from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory

llm = ChatOllama(model="llama3.2", temperature=0.5)

# print(llm.invoke("Hello, how are you?"))

system_prompt = "You are a helpful assistant. \n\n {message}"

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{question}"),
])

llm_with_prompt = prompt | llm

# print(llm_with_prompt.invoke("What is the capital of France?"))


memory = ChatMessageHistory()


while True:
    question = input("User: ")
    response = llm_with_prompt.invoke({"question": question, "message": memory.messages})
    memory.add_user_message(question)
    memory.add_ai_message(response.content)
    print("AI:", response.content)

