from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
from ingest import create_vector_db
 

create_vector_db()
DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
    print("erro aqui 41")
    # Load the locally downloaded model here
    llm = CTransformers(
        
        model = "recogna-nlp/bode-7b-alpaca-pt-br-gguf",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})

    db = FAISS.load_local(DB_FAISS_PATH, embeddings,allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

#chainlit code
import logging

@cl.on_chat_start
async def start():
    try:
        # Initialize the QA bot, potentially loading a model
        chain = qa_bot()
    except Exception as e:
        # Log the error and send an error message to the user if the bot initialization fails
        logging.error(f"Failed to initialize qa_bot: {e}")
        msg = cl.Message(content="Sorry, there was an error starting the Medical Bot. Please try again later.")
        await msg.send()
        return  # Exit the function if the initialization fails

    # Prepare and send the initial message
    initial_msg = cl.Message(content="Starting the bot...")
    await initial_msg.send()

    # Update the message to greet the user and request their query
    initial_msg.content = "Olá! qual a sua pergunta?"
    await initial_msg.update()

    # Store the initialized bot instance in the user session
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    try:
        # Retrieve the bot instance from the user's session
        chain = cl.user_session.get("chain")
        if not chain:
            raise ValueError("Session does not contain a bot instance.")

        # Prepare the callback handler for processing the bot's answer
        cb = cl.AsyncLangchainCallbackHandler(
            stream_final_answer=True,
            answer_prefix_tokens=["FINAL", "ANSWER"]
        )
        cb.answer_reached = True

        # Call the bot with the user's message and handle the response
        res = await chain.acall(message.content, callbacks=[cb])
        answer = res.get("result", "I could not find an answer.")
        sources = res.get("source_documents", [])

        # Append sources to the answer if available
        if sources:
            answer += f"\nSources: {', '.join(sources)}"
        else:
            answer += "\nNo sources found."

        # Send the answer back to the user
        await cl.Message(content=answer).send()
    except Exception as e:
        # Log any errors that occur during message processing
        logging.error(f"Error processing message: {e}")
        error_msg = cl.Message(content="An error occurred while processing your request. Please try again.")
        await error_msg.send()


