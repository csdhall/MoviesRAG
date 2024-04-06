# %%
import time
import json
from dotenv import dotenv_values

import pymongo
from openai import AzureOpenAI

import gradio as gr

# %%
# specify the name of the .env file name
env_name = ".env"  # following example.env template change to your own .env file name
config = dotenv_values(env_name)

cosmos_conn = config["cosmos_connection_string"]
cosmos_database = config["cosmos_database_name"]
cosmos_collection = config["cosmos_collection_name"]
cosmos_vector_property = config["cosmos_vector_property_name"]
cosmos_cache = config["cosmos_cache_collection_name"]
# Create the Azure Cosmos DB for MongoDB client
cosmos_client = pymongo.MongoClient(cosmos_conn)


openai_endpoint = config["openai_endpoint"]
openai_key = config["openai_key"]
openai_version = config["openai_version"]
openai_embeddings_deployment = config["openai_embeddings_deployment"]
openai_embeddings_model = config["openai_embeddings_model"]
openai_embeddings_dimensions = int(config["openai_embeddings_dimensions"])
openai_completions_deployment = config["openai_completions_deployment"]
openai_completions_model = config["openai_completions_model"]
# Create the OpenAI client
openai_client = AzureOpenAI(
    azure_endpoint=openai_endpoint, api_key=openai_key, api_version=openai_version
)


# %%
# Get the database
database = cosmos_client[cosmos_database]

# Get the movie collection
movies = database[cosmos_collection]

# Get the cache collection
cache = database[cosmos_cache]


# %%
import os
from openai import AzureOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt


@retry(wait=wait_random_exponential(min=1, max=200), stop=stop_after_attempt(20))
def generate_embeddings(input_string):
    """
    Retrieves embeddings for the given input string using Azure OpenAI.

    Args:
        input_string (str): The input string for which embeddings need to be retrieved.

    Returns:
        dict: A dictionary containing the response from Azure OpenAI embeddings API.
    """
    client = AzureOpenAI(
        azure_endpoint=os.getenv("openai_endpoint"),
        azure_deployment=os.getenv("openai_embeddings_deployment"),
        api_version=os.getenv("openai_version"),
        api_key=os.getenv("openai_key"),
    )

    response = client.embeddings.create(
        input=input_string, model=os.getenv("openai_embeddings_deployment")
    )

    return response.data[0].embedding


# For testing purposes only
# test = "Led by Woody, Andy's toys live happily in his room until Andy's birthday brings Buzz Lightyear onto the scene. Afraid of losing his place in Andy's heart, Woody plots against Buzz. But when circumstances separate Buzz and Woody from their owner, the duo eventually learns to put aside their differences."
# vectorArray = generate_embeddings(test)
# vectorArray


# %%
def vector_search(collection, vectors, similarity_score=0.02, num_results=5):

    pipeline = [
        {
            "$search": {
                "cosmosSearch": {
                    "vector": vectors,
                    "path": cosmos_vector_property,
                    "k": num_results,
                    "efsearch": 40,  # optional for HNSW only
                },
                "returnStoredSource": True,
            }
        },
        {
            "$project": {
                "similarityScore": {"$meta": "searchScore"},
                "document": "$$ROOT",
            }
        },
        {"$match": {"similarityScore": {"$gt": similarity_score}}},
    ]

    print("Pipeline:\n\n")
    print(pipeline)
    print(f"Collection:{collection}\n\n")

    results = list(collection.aggregate(pipeline))
    print("Results:\n\n")
    for result in results:
        print(result)
    print("Results:\n\n")

    # Exclude the 'vector' to reduce payload size to LLM and _id properties to avoid serialization issues
    for result in results:
        del result["document"]["vector"]
        del result["_id"]
        del result["document"]["_id"]

    return results


# %%
# Grab chat history to as part of the payload to GPT model for completion.
def get_chat_history(completions=3):

    # Sort by _id in descending order and limit the results to the completions value passed in
    results = (
        cache.find({}, {"prompt": 1, "completion": 1})
        .sort([("_id", -1)])
        .limit(completions)
    )

    return results


# %%
def generate_completion(user_prompt, vector_search_results, chat_history):

    system_prompt = """
    You are an intelligent assistant for the Movie Lens Expert AI Assistant.
    You are designed to provide helpful answers to user questions about movies in your database.
    You are friendly, helpful, and informative.
        - Only answer questions related to the information provided below.
        - Write two lines of whitespace between each answer in the list.
        - If you're unsure of an answer, you can say ""I don't know"" or ""I'm not sure"" and recommend users search themselves."
    """

    # Create a list of messages as a payload to send to the OpenAI Completions API

    # system prompt
    messages = [{"role": "system", "content": system_prompt}]

    # chat history
    for chat in chat_history:
        messages.append(
            {"role": "user", "content": chat["prompt"] + " " + chat["completion"]}
        )

    # user prompt
    messages.append({"role": "user", "content": user_prompt})

    # vector search results
    for result in vector_search_results:
        messages.append({"role": "system", "content": json.dumps(result["document"])})

        # Create the completion
        response = openai_client.chat.completions.create(
            model=openai_completions_deployment, messages=messages
        )

    return response.model_dump()


# %%
def cache_response(user_prompt, prompt_vectors, response):

    chat = [
        {
            "prompt": user_prompt,
            "completion": response["choices"][0]["message"]["content"],
            "completionTokens": str(response["usage"]["completion_tokens"]),
            "promptTokens": str(response["usage"]["prompt_tokens"]),
            "totalTokens": str(response["usage"]["total_tokens"]),
            "model": response["model"],
            cosmos_vector_property: prompt_vectors,
        }
    ]

    cache.insert_one(chat[0])


# %%
def chat_completion(user_input):

    # Generate embeddings from the user input
    user_embeddings = generate_embeddings(user_input)

    # Query the chat history cache first to see if this question has been asked before
    # Similarity score set to 0.99, will only return exact matches. Limit to 1 result.
    cache_results = vector_search(
        cache, user_embeddings, similarity_score=0.99, num_results=1
    )

    if len(cache_results) > 0:

        return cache_results[0]["document"]["completion"]

    else:

        # perform vector search on the movie collection
        search_results = vector_search(movies, user_embeddings)

        # chat history
        chat_history = get_chat_history(3)

        # generate the completion
        completions_results = generate_completion(
            user_input, search_results, chat_history
        )

        # cache the response
        cache_response(user_input, user_embeddings, completions_results)

        # Return the generated LLM completion
        return completions_results["choices"][0]["message"]["content"]


# %%
chat_history = []
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Ask me anything about movies!")
    clear = gr.Button("Clear")

    def user(user_message, chat_history):

        # Create a timer to measure the time it takes to complete the request
        start_time = time.time()

        # Get LLM completion
        response_payload = chat_completion(user_message)

        # Stop the timer
        end_time = time.time()

        elapsed_time = round((end_time - start_time) * 1000, 2)

        response = response_payload

        # Append user message and response to chat history
        chat_history.append(
            [user_message, response_payload + f"\n (Time: {elapsed_time}ms)"]
        )

        return gr.update(value=""), chat_history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False)

    clear.click(lambda: None, None, chatbot, queue=False)

# %%

# launch the gradio interface
demo.launch(debug=True)


# %%
# be sure to run this cell to close or restart the gradio demo
demo.close()
