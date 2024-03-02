from langchain_community.vectorstores import Chroma
import openai
from langchain_openai import OpenAIEmbeddings
import os
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from langchain.agents import Tool
from langchain.agents import initialize_agent


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your specific needs in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def chat_response(param):
    embeddings = OpenAIEmbeddings(openai_api_key="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")
    persist_directory = 'chroma_db'
    new_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    os.environ['OPENAI_API_KEY'] = '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'

    model_name = 'gpt-3.5-turbo'
    llm = ChatOpenAI(model_name=model_name, temperature=0.1)

    query = f"{param}"
    retrieval_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=new_db.as_retriever())
    # res = retrieval_chain.invoke(query)

    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
    )


    from langchain import PromptTemplate


    prompt = PromptTemplate(
        template="""Kamu adalah bot assisten help center ipb, civitas ipb ingin bertanya mengenai/
        {pertanyaan}. Berikanlah jawaban yang benar, sopan dan membantu.""",
        input_variables=["pertanyaan"]
    )

    # format the prompt to add variable values
    prompt_formatted_str: str = prompt.format(
        pertanyaan=query)

    tools = [
        Tool(
            name="Dasar pengetahuan",
            func=retrieval_chain.run,
            description=("gunakan alat ini saat menjawab pertanyaan pengetahuan umum untuk mendapatkan informasi "
                         "lebih lanjut tentang topik tersebut, jawab dengan bahasa indonesia")
        )
    ]

    agent = initialize_agent(
        agent="conversational-react-description",
        tools=tools,
        llm=llm,
        verbose=False,
        max_iterations=5,
        early_stopping_method="generate",
        memory=conversational_memory
    )

    res = agent.invoke(prompt_formatted_str)
    print(res)
    return res



@app.get("/")
async def root(param: str):
    print(param)
    res = chat_response(param)
    print(res)
    return res

