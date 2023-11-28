from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, TypeAlias, cast

from multimethod import multimethod


class MessageRole(StrEnum):
    HUMAN = "human"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class IntoMessage(Protocol):
    def into_message(self) -> Message:
        ...


@dataclass
class Message(IntoMessage):
    """
    A message to be sent to a model.
    """
    role: MessageRole
    content: str
    extra: Any = None

    def into_message(self) -> Message:
        return self


@multimethod
def dispatch_into_message(message: str):  # type: ignore
    return Message(MessageRole.HUMAN, message)


@multimethod
def dispatch_into_message(message: tuple[MessageRole, object]):
    return Message(role=message[0], content=str(message[1]))


@dispatch_into_message.register
def _(message):
    return cast(IntoMessage, message).into_message()


SomeMessage: TypeAlias = "str | tuple[MessageRole, object] | IntoMessage"


def into_message(message: SomeMessage) -> Message:
    return dispatch_into_message(message)  # type: ignore


__all__ = ["Message", "MessageRole", "IntoMessage", "into_message", "SomeMessage"]
