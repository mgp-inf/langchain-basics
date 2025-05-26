"""Demonstrates a Gemini API call using LangChain messages with ChatTemplate."""
import os
import argparse
import langchain_google_genai as lc_genai
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.prompts.chat import AIMessagePromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
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

    # create a list of messages
    system_template = "You are a {persona}. You try to {persona_task}."
    question_template = "What is the meaning of {word}?"
    system_prompt = SystemMessagePromptTemplate.from_template(system_template)
    question_prompt = HumanMessagePromptTemplate.from_template(question_template)
    messages_prompt = ChatPromptTemplate.from_messages([system_prompt, question_prompt])
    persona = "compassionate helper and does your best to provide answer to any question"
    persona_task = (
        "make sense of the question breaking it down into parts "
        "and then evaluating them and explaining them to the user"
    )
    word = "frahanitrateprerajulization"
    messages = messages_prompt.format_prompt(persona=persona, persona_task=persona_task, word=word)

    # run a prompt
    result = llm.invoke(messages)
    print(f"---\nSystem: {system_prompt.format(persona=persona, persona_task=persona_task).content}\n---")
    print(f"You: {question_prompt.format(word=word).content}")
    print(f"AI: {result.content}")


if __name__ == "__main__":
    args = get_args()
    main(args.model, args.temperature)
