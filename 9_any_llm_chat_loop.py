"""Demonstrates any LLM API call using LangChain and use of a tool with an interactive chat loop."""
import os
import argparse
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage
import datetime
import logging
from dotenv import load_dotenv

# load GOOGLE_API_KEY
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPEN_API_KEY = os.getenv("GOOGLE_API_KEY")

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


def get_provider(model_name: str) -> str:
    """Given a model name, returns the provider used.

    Args:
        model_name: Name of the model.

    Raises:
        KeyError: When the name doesn't match any of the available providers.
    """
    if "gpt" in model_name.lower():
        return "openai"
    elif "gemini" in model_name.lower():
        return "google_genai"
    elif "claude" in model_name.lower():
        return "anthropic"
    else:
        raise KeyError("Currently detects only OpenAI, Gemini and Anthropic models")


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
        provider = get_provider(model_name)
        llm = init_chat_model(model=model_name, model_provider=provider, temperature=temperature)
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
        # "If you know the answer use the default arguments needed."
    )
    question = "What is the current time?"
    messages = [
        SystemMessage(system_message),
        HumanMessage(question),
    ]
    
    # run a prompt
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        messages.append(HumanMessage(user_input))
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        tools_called = False
        for tool_call in response.tool_calls: 
            selected_tool = tools_map[tool_call["name"]]
            messages.append(selected_tool.invoke(tool_call))
            tools_called = True
        if tools_called:
            response = llm_with_tools.invoke(messages)
            messages.append(response)
        print(f"\nAI: {response.content}")


if __name__ == "__main__":
    args = get_args()
    main(args.model, args.temperature)
