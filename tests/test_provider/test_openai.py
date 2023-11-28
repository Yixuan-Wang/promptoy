from dotenv import load_dotenv

load_dotenv()

import pytest

def test_message_role():
    from promptoy.interface.message import MessageRole, Message, into_message
    tuple_messages = [
        (MessageRole.HUMAN, "Hello, world!"),
        (MessageRole.ASSISTANT, "Hello, world! too."),
    ]

    messages = [
        Message(MessageRole.HUMAN, "Hello, world!"),
        Message(MessageRole.ASSISTANT, "Hello, world! too."),
    ]

    for src, target in zip(tuple_messages, messages):
        assert into_message(src) == target

def test_openai_message():
    from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam
    from pydantic import TypeAdapter, ValidationError

    from promptoy.interface.message import MessageRole, into_message
    from promptoy.provider.openai import EndpointOpenAI

    messages = [
        (MessageRole.HUMAN, "Hello, world!"),
        (MessageRole.ASSISTANT, "Hello, world! too."),
        (MessageRole.SYSTEM, "You are an assistant echoing the human in a friendly and respectful manner."),
    ]
    messages = EndpointOpenAI.format_message(into_message(m) for m in messages)

    message_models = [
        ChatCompletionUserMessageParam,
        ChatCompletionAssistantMessageParam,
        ChatCompletionSystemMessageParam,
    ]

    for src, model in zip(messages, message_models):
        try:
            TypeAdapter(model).validate_python(src)
        except ValidationError as e:
            pytest.fail(f"Failed to validate {src} against {model}: {e}")

