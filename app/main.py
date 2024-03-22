import json
import os
import ssl
from datetime import datetime
from select import select
from shutil import move, rmtree

import nltk
from databases import Database
from fastapi import FastAPI, Request, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.chains import RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.document_loaders import DirectoryLoader
from langchain.memory import PostgresChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from sqlalchemy import create_engine, Column, Integer, String, MetaData, Table, DateTime, Boolean, desc

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')

app = FastAPI()
public_folder = os.getcwd() + "/public"
training_material_folder = os.getcwd() + "/training_data"
persist_directory = os.getcwd() + '/chroma_db'

os.environ['DATABASE_URL'] = "postgresql://llm:llm@localhost/llm2"
os.environ['OPENAI_API_KEY'] = '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'

database = Database(os.environ.get('DATABASE_URL'))

metadata = MetaData()

# database schema
files_table = Table(
    "uploaded_files",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("original_filename", String),
    Column("size", Integer),
    Column("unique_filename", String),
    Column("is_trained", Boolean),
    Column('trained_at', DateTime, default=None)
)

training_history_table = Table(
    "training_history",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("trained_at", DateTime),
    Column("files", String),
    Column("total_token", Integer),
    Column('is_success', Boolean, default=None)
)

engine = create_engine(os.environ.get('DATABASE_URL'), pool_pre_ping=True)
metadata.create_all(engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your specific needs in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def chat_response(param, sessionKey, queryDetail):
    global full_path_most_recent

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get('OPENAI_API_KEY'))
    persist_directory = 'chroma_db'

    all_files = os.listdir(persist_directory)

    # Filter only files with a specific naming pattern (assuming they are named as dates in string format)
    date_files = [file for file in all_files if
                  file.isdigit()]  # You may need to adjust this depending on your naming convention

    most_recent_file = max(date_files, key=int)
    full_path_most_recent = persist_directory + '/' + most_recent_file

    new_db = Chroma(persist_directory=full_path_most_recent, embedding_function=embeddings)

    model_name = 'gpt-3.5-turbo'
    llm = ChatOpenAI(model_name=model_name, temperature=0.1)

    query = f"{param}"
    retrieval_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=new_db.as_retriever())
    # res = retrieval_chain.invoke(query)

    # connection_string = ("mongodb+srv://llmuser:Llm2805200431052004@cluster0.k6dosmb.mongodb.net/?retryWrites=true&w"
    #                      "=majority&appName=Cluster0")

    chat_history =  PostgresChatMessageHistory(
        connection_string=os.environ.get('DATABASE_URL'),
        session_id=f"{sessionKey}",
    )

    conversational_memory = ConversationBufferWindowMemory(
        chat_memory=chat_history,
        memory_key = 'chat_history',
        return_messages = True,
        # output_key="answer"
    )

    # conversational_memory = ConversationBufferWindowMemory(
    #     memory_key='chat_history',
    #     k=5,
    #     return_messages=True
    # )



    # prompt template
    from langchain.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )

    from langchain.prompts import PromptTemplate

    sys_prompt: PromptTemplate = PromptTemplate(
        input_variables=[],
        template="""kamu seorang ai chatbot assisten help center di ipb university (institut pertanian bogor), 
        yang membantu para civitas ipb dengan suka hati dan sangatlah ramah. 
        Bantulah mereka dengan memberikan respon jawaban yang selalu benar, sopan dan membantu. Terus tawarkan ke 
        para civitas ipb 
        adakah 
        yang 
        bisa dibantu 
        kembali. Dan jika semisalnya diskusi dengan user tidak mencapai titik temu, tawarkanlah bantuan untuk diarahkan 
        agar 
        berbicara langsung dengan pihak manusia asli yang bekerja di help center"""
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt=sys_prompt)

    judul = queryDetail['judul']
    deskripsi = queryDetail['deskripsi']
    topik = queryDetail['topik']
    nama = queryDetail['nama']

    sys_prompt2: PromptTemplate = PromptTemplate(
        input_variables=["judul", "deskripsi", "topik", "nama"],
        template=f"""
            User ini bernama: {nama}
            Memiliki pengaduan berjudul: {judul}
            Deskripsi pengaduannya adalah: {deskripsi}
            Topik dari pengaduan tersebut ialah: {topik}
            Selalu jawablah dengan menggunakan bahasa indonesia!
        """
    )

    system_message_prompt2 = SystemMessagePromptTemplate(prompt=sys_prompt2)


    student_prompt: PromptTemplate = PromptTemplate(
        input_variables=["query"],
        template="{query}"
    )
    student_message_prompt = HumanMessagePromptTemplate(prompt=student_prompt)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, system_message_prompt2, student_message_prompt])

    chat_prompt_format = chat_prompt.format_messages(
        query=query,
    )
    print(chat_prompt_format)

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

    res = agent.invoke(chat_prompt_format)

    chat_history.add_user_message(param)
    chat_history.add_ai_message(res['output'])
    print(res)
    return res


def split_docs(documents, chunk_size=1000, chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

async def move_to_train(unique_filename):
    try:
        # Set the paths for the source (public folder) and destination (training material folder)
        source_path = os.path.join(public_folder, unique_filename)
        destination_path = os.path.join(training_material_folder, unique_filename)

        # Check if the file exists in the public folder
        if os.path.exists(source_path):
            # Move the file to the training material folder
            move(source_path, destination_path)
            return True

        else:
            raise HTTPException(status_code=404, detail=f"File '{unique_filename}' not found in public folder.")

    except Exception as e:
        # Handle exceptions here
        raise HTTPException(status_code=500, detail=str(e))

async def move_to_public(unique_filename):
    try:
        # Set the paths for the source (public folder) and destination (training material folder)
        source_path = os.path.join(training_material_folder, unique_filename)
        destination_path = os.path.join(public_folder, unique_filename)

        # Check if the file exists in the public folder
        if os.path.exists(source_path):
            # Move the file to the training material folder
            move(source_path, destination_path)
            return True

        else:
            raise HTTPException(status_code=404, detail=f"File '{unique_filename}' not found in public folder.")

    except Exception as e:
        # Handle exceptions here
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/")
async def root(request: Request):
    # Access the param from the request body
    request_param = await request.json()
    print(request_param)

    # Assuming `chat_response` is a function that processes the request_param
    res = chat_response(request_param['question'], request_param['sessionKey'], request_param['queryDetail'])
    print(res)

    return res

@app.post('/add-parameter')
async def add_parameter(file: UploadFile):

    try:
        # Check if the file type is PDF
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        unique_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{file.filename}"

        # Store in a public folder
        file_path = os.path.join(public_folder, unique_filename)
        with open(file_path, 'wb') as f:
            f.write(file.file.read())

        async with database.transaction():
            query = files_table.insert().values(
                original_filename=file.filename,
                size=file.file.seek(0, os.SEEK_END),
                unique_filename=unique_filename,
                is_trained=False,
                trained_at=None
            )
            await database.execute(query)

        # Return success message
        return {"message": "File berhasil diinput", "status_code": 201, "success": True}

    except Exception as e:
        # Handle other exceptions here
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/parameters")
async def get_parameter():
    try:
        query = files_table.select()
        parameters = await database.fetch_all(query)

        return parameters

    except Exception as e:
        # Handle other exceptions here
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/activate-parameter")
async def activate_parameter(request: Request):
    try:
        request_param = await request.json()
        print(request_param)

        fileId = int(request_param['fileId'])
        mode = request_param['mode']

        query = files_table.select().where(files_table.c.id == fileId)
        parameter = await database.fetch_one(query)

        if(mode == 'train'):
            res = await move_to_train(parameter['unique_filename'])

            if(res == True):
                update_query = (
                    files_table.update()
                    .where(files_table.c.id == fileId)
                    .values(trained_at=datetime.now(), is_trained=True)
                )
                await database.execute(update_query)

            return f"File {parameter['original_filename']} Berhasil dimasukkan ke dalam folder training"

        elif(mode == 'untrain'):
            res = await move_to_public(parameter['unique_filename'])

            if (res == True):
                update_query = (
                    files_table.update()
                    .where(files_table.c.id == fileId)
                    .values(trained_at=None, is_trained=False)
                )
                await database.execute(update_query)

            return f"File {parameter['original_filename']} Berhasil dihapus dari folder training"

    except Exception as e:
        # Handle other exceptions here
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/train-model")
async def train_model():
   try:
       loader = DirectoryLoader(os.path.join(training_material_folder))
       documents = loader.load()

       chunk_size = 1000
       chunk_overlap = 20

       text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
       docs = text_splitter.split_documents(documents)

       async with database.transaction():
           file_names = os.listdir(training_material_folder)
           files_concatenated = ', '.join(file_names)

           query = training_history_table.insert().values(
               trained_at=datetime.now(),
               files=files_concatenated,
               total_token=len(docs),
               is_success=None,
           )
           inserted_id = await database.execute(query)

       embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get('OPENAI_API_KEY'))
       # db = Chroma.from_documents(docs, embeddings)
       #
       vectordb = Chroma.from_documents(
           documents=docs, embedding=embeddings, persist_directory=os.path.join(persist_directory, datetime.now(
           ).strftime("%Y%m%d%H%M%S"))
       )

       # rmtree(persist_directory, ignore_errors=True)
       #
       # vectordb = Chroma.from_documents(
       #     documents=docs, embedding=embeddings, persist_directory=persist_directory)
       #
       # vectordb.persist()

       # Update the is_success column
       update_query = training_history_table.update().where(
           training_history_table.c.id == inserted_id
       ).values(is_success=True)

       await database.execute(update_query)

       return f"Training Model Berhasil"
   except Exception as e:
        # Handle other exceptions here
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/training-history")
async def training_history():
    try:
        query = training_history_table.select().order_by(desc("trained_at"))
        parameters = await database.fetch_all(query)

        return parameters

    except Exception as e:
        # Handle other exceptions here
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/delete-file/{file_id}")
async def delete_file(file_id: int):
    try:
        # Use the select query without passing a list to select method
        query = files_table.select().where(files_table.c.id == file_id)
        result = await database.fetch_one(query)

        if result:
            uf = result['unique_filename']

            if result['is_trained']:
                raise HTTPException(status_code=404, detail="File still in use for training")

            file_path = os.path.join(public_folder, uf)

            if os.path.exists(file_path):
                try:
                    # Use a context manager to open and close the file
                    with open(file_path, 'rb'):
                        os.remove(file_path)
                except Exception as delete_error:
                    raise HTTPException(status_code=500, detail=f"Error deleting file: {str(delete_error)}")

                # Use the delete query without passing a list to where method
                delete_query = files_table.delete().where(files_table.c.id == file_id)
                await database.execute(delete_query)

                return {"message": "File deleted successfully"}

            raise HTTPException(status_code=404, detail="File not found")

        raise HTTPException(status_code=404, detail="File ID not found")

    except HTTPException:
        # Re-raise HTTPExceptions to let FastAPI handle them
        raise

    except Exception as e:
        # Handle other exceptions here
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_db_client():
    await database.connect()

@app.on_event("shutdown")
async def shutdown_db_client():
    await database.disconnect()
