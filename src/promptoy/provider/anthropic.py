from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional, Never, TypedDict, NotRequired, Unpack

from anthropic import Anthropic

from promptoy.interface.chat import MakeChat, Chat
from promptoy.interface.message import MessageRole, SomeMessage, into_message


class ModelAnthropic(StrEnum):
    CLAUDE = "claude-2"
    CLAUDE_INSTANT = "claude-instant"


class ParamsAnthropic(TypedDict):
    model: ModelAnthropic | str
    temperature: NotRequired[Optional[float]]
    max_tokens: NotRequired[int]


@dataclass
class EndpointAnthropic(
    MakeChat[ParamsAnthropic, Never],
):
    client: Optional[Anthropic] = field(default=None)

    def get_client(self) -> Anthropic:
        if self.client is None:
            self.client = Anthropic()
        return self.client

    @staticmethod
    def __format_one_message(message: SomeMessage):
        message = into_message(message)
        match message.role:
            case MessageRole.HUMAN:
                return dict(role="human", content=message.content)
            case MessageRole.ASSISTANT:
                return dict(role="assistant", content=message.content)
            case MessageRole.SYSTEM:
                return dict(role="human", content=message.content)
            case MessageRole.TOOL:
                # TODO[tool]
                raise NotImplementedError("Tools are not yet implemented.")
            case _:
                return dict(role="human", content=message.content)

    @staticmethod
    def format_message(
            messages: Iterable[SomeMessage],
            *,
            validate: bool = False,
            auto_pad: bool = True,
    ):
        """
        Format messages into Anthropic's format (`"Human:"` and `"Assistant:"` string).

        [Anthropic's requirements on prompts](https://docs.anthropic.com/claude/reference/prompt-validation) are:
        - The first conversational turn in the prompt must be a `"Human:"` turn.
        - The last conversational turn in the prompt be an `"Assistant:"` turn.
        - The prompt must be less than 100,000 - 1 tokens in length.

        Args:
            messages (Iterable[SomeMessage]): Iterable of standard `Message`s.
            validate (bool, optional): Whether to validate the messages against Anthropic's requirements. Defaults to False.
            auto_pad (bool, optional): Whether to pad messages to fit the format and skip validation. Defaults to True.

        Exceptions:
            ValueError: If `validate=True` and the messages do not meet Anthropic's requirements.
        """

        anthropic_messages = list(map(EndpointAnthropic.__format_one_message, messages))
        if len(anthropic_messages) == 0:
            raise ValueError("Messages must not be empty.")

        if anthropic_messages[0]["role"] != "human":
            if auto_pad:
                anthropic_messages.insert(0, dict(role="human", content="Hello."))
            elif validate:
                raise ValueError("The first message must be from the human.")

        if anthropic_messages[-1]["role"] != "assistant":
            if auto_pad:
                anthropic_messages.append(dict(role="assistant", content=""))
            elif validate:
                raise ValueError("The last message must be from the assistant.")

        prompt = "".join(f"\n\n{m['role'].capitalize()}: {m['content']}" for m in anthropic_messages)

        return prompt

    def make_chat(
            self,
            **params: Unpack[ParamsAnthropic],
    ) -> Chat[Never]:
        def _(messages: Iterable[SomeMessage], **ctx: Never) -> str:
            prompt = self.format_message(messages, validate=True)
            return self.get_client().completions.create(
                prompt=prompt,
                model=str(params["model"]),
                stream=False,
                max_tokens_to_sample=params.get("max_tokens") or 1024,  # type: ignore
                temperature=params.get("temperature") or 1.0,  # type: ignore
            ).completion

        return _
