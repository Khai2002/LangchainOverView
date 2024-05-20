# Set-Up ==================================================================================
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from typing import List, Optional, TypedDict
import uuid
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

load_dotenv()

# === Global Variables ====================================================================

llm = ChatOpenAI(temperature=1, model="gpt-3.5-turbo-0613", max_tokens=512)

anime_list = pd.read_parquet('anime/anime.parquet')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
matrix = joblib.load('sparse_matrix.pkl')
num_samples = 50
num_anime_rec = 3


# === Classes =============================================================================


class Anime(BaseModel):
    """
    Information about an Anime that the user likes
    """

    description: Optional[str] = Field(default=None, description="The description of the anime")


class Data(BaseModel):
    """
    Extracted data about anime into a list of 2 element
    """

    list: List[Anime]


class Example(TypedDict):
    """A representation of an example consisting of text input and expected tool calls.

    For extraction, the tool calls are represented as instances of pydantic model.
    """

    input: str  # This is the example text
    tool_calls: List[BaseModel]  # Instances of pydantic model that should be extracted


# === Util Functions ======================================================================

def recommend_anime_fictional_syn(data):
    synopsis = [anime.description for anime in data.list]
    tfidf_matrix_new = vectorizer.transform(synopsis)
    cosine_sim = linear_kernel(tfidf_matrix_new, matrix)
    average_similarities = np.mean(cosine_sim, axis=0)
    sim_scores = list(enumerate(average_similarities))
    sorted_sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    indices = [idx for idx, sim in sorted_sim_scores[:num_samples]]
    selected_animes = anime_list.loc[indices]
    filtered_df = selected_animes[selected_animes['Type'] != 'Music']
    sorted_anime_list = filtered_df.sort_values('Score', ascending=False).iloc[:num_anime_rec]
    return sorted_anime_list['Name'].to_list()


def list_to_string(ls):
    rep = "Recommend me these Animes: " + ls[0]
    for s in ls[1:]:
        rep += ", "
        rep += s
    rep += ". Give short reason for each recommendation."
    return rep


def tool_example_to_messages(example: Example) -> List[BaseMessage]:
    """Convert an example into a list of messages that can be fed into an LLM.

    This code is an adapter that converts our example to a list of messages
    that can be fed into a chat model.

    The list of messages per example corresponds to:

    1) HumanMessage: contains the content from which content should be extracted.
    2) AIMessage: contains the extracted information from the model
    3) ToolMessage: contains confirmation to the model that the model requested a tool correctly.

    The ToolMessage is required because some of the chat models are hyper-optimized for agents
    rather than for an extraction use case.
    """
    messages: List[BaseMessage] = [HumanMessage(content=example["input"])]
    openai_tool_calls = []
    for tool_call in example["tool_calls"]:
        openai_tool_calls.append(
            {
                "id": str(uuid.uuid4()),
                "type": "function",
                "function": {
                    # The name of the function right now corresponds
                    # to the name of the pydantic model
                    # This is implicit in the API right now,
                    # and will be improved over time.
                    "name": tool_call.__class__.__name__,
                    "arguments": tool_call.json(),
                },
            }
        )
    messages.append(
        AIMessage(content="", additional_kwargs={"tool_calls": openai_tool_calls})
    )
    tool_outputs = example.get("tool_outputs") or [
        "You have correctly called this tool."
    ] * len(openai_tool_calls)
    for output, tool_call in zip(tool_outputs, openai_tool_calls):
        messages.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))
    return messages


# === Reference Examples ==================================================================

# Defining examples
examples = [
    (
        "I want an anime about camping.",
        Data(list=[
            Anime(
                description='A group of high school students go on a camping trip and learn about nature and '
                            'friendship.'),
            Anime(
                description='A girl goes on a solo camping trip to find solace and discover herself in the great '
                            'outdoors.')
        ])
    ),
    (
        "I want a romantic anime for girls.",
        Data(list=[
            Anime(
                description='A heartwarming story about two high school students who fall in love and navigate the '
                            'ups and downs of their relationship.'),
            Anime(
                description='A girl moves to a new town and finds herself drawn to a mysterious boy with a tragic '
                            'past. Together, they discover the power of love and healing.')
        ])
    ),
    (
        "What is 2+2?",
        Data(list=[])
    ),
    (
        "Give me a fun fact",
        Data(list=[])
    ),

]

messages = []
for text, tool_call in examples:
    messages.extend(
        tool_example_to_messages({"input": text, "tool_calls": [tool_call]})
    )

# === Prompt Templates ====================================================================


classification_prompt = ChatPromptTemplate.from_template(
    """
            Given the user request below, classify it as either being about `Anime or Story` or `Other`.
            
            Be more willing to classify `Anime or Story`
            Do not respond with more than one word.
            
            <request>
            {request}
            </request>
            
            Classification:
        """
)

anime_synopsis_generator_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "You only extract a maximum of 2 elements in the list"
            "You will need to extract a very short description of the anime without giving its name."
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value."
            ,
        ),
        MessagesPlaceholder('examples'),
        ("human", "{request}"),
    ]
)

general_prompt = ChatPromptTemplate.from_template(
    """Respond to the following question:

    Request: {request}
    Answer:
    """
)

# === Sub-Chains =======================================================================

chain = (
        classification_prompt
        | llm
        | StrOutputParser()
)

anime_chain = (
        anime_synopsis_generator_prompt
        | llm.with_structured_output(schema=Data)
        | RunnableLambda(recommend_anime_fictional_syn)
        | RunnableLambda(list_to_string)
        | llm
        | StrOutputParser()
)

general_chain = (
        general_prompt
        | llm
)


# === Chain Routing ===================================================================


def route(info):
    if "anime" in info["topic"].lower():
        return anime_chain
    else:
        return general_chain


# === Full Chain ======================================================================


full_chain = {"topic": chain, "request": lambda x: x["request"], "examples": lambda x: x["examples"]} | RunnableLambda(
    route
)

# This is usage, the others should be init
print(full_chain.invoke({"request": "recommend me an adventure anime", "examples": messages}))
