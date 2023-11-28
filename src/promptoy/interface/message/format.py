from . import SomeMessage, into_message, MessageRole


def format_one_message_into_openai(message: SomeMessage):
    message = into_message(message)
    match message.role:
        case MessageRole.HUMAN:
            return dict(role="user", content=message.content)
        case MessageRole.ASSISTANT:
            return dict(role="assistant", content=message.content)
        case MessageRole.SYSTEM:
            return dict(role="system", content=message.content)
        case MessageRole.TOOL:
            # TODO[tool]
            raise NotImplementedError("Tools are not yet implemented.")
        case _:
            return dict(role="user", content=message.content, name=message.role)
