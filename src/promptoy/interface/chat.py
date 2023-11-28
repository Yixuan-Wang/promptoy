from __future__ import annotations

from typing import Protocol, TypeVar
from collections.abc import Iterable

from promptoy.interface.message import SomeMessage

C = TypeVar("C", contravariant=True)
P = TypeVar("P", contravariant=True)


class Chat(Protocol[C]):
    def __call__(self, messages: Iterable[SomeMessage], **ctx: C) -> str:
        ...


class MakeChat(Protocol[P, C]):
    def make_chat(self, **params: P) -> Chat[C]:
        ...

__all__ = ["Chat", "MakeChat"]
