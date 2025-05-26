"""Demonstrates a Gemini API call using LangChain using a tool."""
import os
import argparse
import langchain_google_genai as lc_genai
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage
import datetime
import logging
from dotenv import load_dotenv

# load GOOGLE_API_KEY
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def get_args() -> argparse.Namespace:
    """Parses the input arguments and returns the object holding these attributes.

    Returns:
        argparse.Namespace object holding the parsed arguments as attributes.
    """
    # 1. Create the parser
    parser = argparse.ArgumentParser(
        description=(
            "A simple conversational AI assistant "
            "capable of retaining context across turns."
        ),
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    # 2. Add arguments
    parser.add_argument("--model", type=str, default="gemini-2.0-flash", help="The model to be used")
    parser.add_argument("--temperature", type=float, default=0.7, help="The temperature parameter for the model")
    # 3. Parse the arguments
    args = parser.parse_args()
    return args


@tool
def get_system_time(format: str = "%d-%m-%Y %H:%M:%S") -> str:
    """Given the time format, returns the system time.

    Args:
        format: The string indicating the needed time format.

    Returns:
        The detected time in the required format.
    """
    current_time = datetime.datetime.now()
    output = current_time.strftime(format)
    return output


def main(model_name: str, temperature: float) -> None:
    """Given the model name and temperature parameter, this initiates an interactive chat.

    Args:
        model_name: Name of the Gemini model.
        temperature: The temperature parameter of the model.

    Raises:
        Exception: If the model fails to load or execute a given prompt.
    """
    # load model
    try:
        llm = lc_genai.ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        logging.info("Successfully loaded '%s'!", model_name)
    except Exception as e:
        logging.error("Failed to load '%s'!\n%s", model_name, e)
        raise

    # create the model binding to the tool
    llm_with_tools = llm.bind_tools([get_system_time])
    tools_map = {"get_system_time": get_system_time}

    # ask time
    system_message = (
        "Explicitly infrom the user whether or not you are able to find the answer. "
        "If not, say you don't know the answer, but inform what steps the user can take to get the answer. "
        "If you know the answer use the default arguments needed."
    )
    question = "What is the current time?"
    messages = [
        SystemMessage(system_message),
        HumanMessage(question),
    ]
    
    # run a prompt
    result = llm.invoke(messages)
    result_with_tools = llm_with_tools.invoke(messages)
    tool_messages = []
    tools_used = []
    for tool_call in result_with_tools.tool_calls: 
        selected_tool = tools_map[tool_call["name"]]
        tool_messages.append(selected_tool.invoke(tool_call))
        tools_used.append(tool_call["name"] + "()")
    all_messages = messages + [result_with_tools] + tool_messages
    output_with_tools = llm_with_tools.invoke(all_messages)
    print(f"You: {question}")
    print(f"AI without tools: {result.content}")
    print(f"AI with tools: {output_with_tools.content}")
    print(f"Used the following tools: {tools_used}")


if __name__ == "__main__":
    args = get_args()
    main(args.model, args.temperature)
