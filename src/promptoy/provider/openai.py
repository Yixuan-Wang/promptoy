from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Never, NotRequired, Optional, TypedDict, Unpack

from openai import OpenAI as OpenAIClient

from promptoy.interface.chat import Chat, MakeChat
from promptoy.interface.message import SomeMessage
from promptoy.interface.message.format import format_one_message_into_openai


class ModelOpenAI(str, Enum):
    GPT_3_5_TURBO = "gpt-3.5-turbo-1106"
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-1106-preview"


class ParamsOpenAI(TypedDict):
    model: ModelOpenAI | str
    temperature: NotRequired[Optional[float]]


@dataclass(kw_only=True)
class EndpointOpenAI(
    MakeChat[ParamsOpenAI, Never],
):
    """
    An endpoint for calling OpenAI provider.
    
    See Also:
        [Original API](https://platform.openai.com/docs/api-reference/chat)
    """

    client: Optional[OpenAIClient] = field(default=None)
    """A lazily initialized OpenAI client."""

    # TODO[async]
    # async_client: Optional[AsyncOpenAIClient] = field(default=None)

    def get_client(self) -> OpenAIClient:
        if self.client is None:
            self.client = OpenAIClient()
        return self.client

    # TODO[async]
    # def _get_async_client(self) -> AsyncOpenAIClient:
    #     if self.async_client is None:
    #         self.async_client = AsyncOpenAIClient()
    #     return self.async_client

    @staticmethod
    def format_message(messages: Iterable[SomeMessage]):
        """
        Format messages into OpenAI's API format (an array of maps).

        See Also:
            [Original API](https://platform.openai.com/docs/api-reference/chat/create)

        Args:
            messages (Iterable[TIntoMessage]): Iterable of standard `Message`s.
        """

        return list(map(format_one_message_into_openai, messages))

    def make_chat(
        self,
        **params: Unpack[ParamsOpenAI],
    ) -> Chat[Never]:
        def _(messages: Iterable[SomeMessage], **ctx: Never) -> str:
            choices = self.get_client().chat.completions.create(
                **{
                    **params,
                    "model": str(params["model"]),
                    "messages": self.format_message(messages),  # type: ignore
                    "temperature": params.get("temperature"),
                }
            ).choices

            if len(choices) == 0:
                return ""

            return choices[0].message.content or ""

        return _

    # TODO[async]
    # def _get_chat_async(self,
    #     model: ModelOpenAI | str,
    #     temperature: float = 1,
    # ):
    #     async def _(messages: Iterable[TIntoMessage]):
    #         choices = (await self._get_async_client().chat.completions.create(
    #             model=model,
    #             temperature=temperature,
    #             messages=self.format_message(messages), # type: ignore
    #         )).choices

    #         if len(choices) == 0:
    #             return ""

    #         return choices[0].message.content or ""

    #     return _
