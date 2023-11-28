from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from os import PathLike
import os
from typing import Never, NotRequired, Optional, TypedDict, Unpack

from llama_cpp import Llama

from promptoy.interface.chat import Chat, MakeChat
from promptoy.interface.message import SomeMessage
from promptoy.interface.message.format import format_one_message_into_openai


class ParamsLlamaCpp(TypedDict):
    model: PathLike | str | bytes
    temperature: NotRequired[Optional[float]]
    gpu_layers: NotRequired[int]


@dataclass(kw_only=True)
class EndpointLlamaCpp(
    MakeChat[ParamsLlamaCpp, Never]
):
    llms: dict[str, Llama] = field(default_factory=dict)

    @staticmethod
    def format_message(messages: Iterable[SomeMessage]):
        """
        Format messages into LLaMA's API format (an array of maps), which is the same
        as OpenAI's latest API format.
        """

        return list(map(format_one_message_into_openai, messages))

    def make_chat(self, **params: Unpack[ParamsLlamaCpp]) -> Chat[Never]:
        """
        Make a chat with LLaMA.
        """
        model = str(params["model"])
        if model not in self.llms:
            self.llms[model] = Llama(
                model_path=str(os.fspath(params["model"])),
                n_gpu_layers=params.get("gpu_layers", 0),
            )
        llm = self.llms[model]

        def chat(messages: Iterable[SomeMessage], **ctx: Never) -> str:
            completion = llm.create_chat_completion(
                messages=self.format_message(messages),  # type: ignore
                temperature=params.get("temperature") or 0.2,
            )

            return completion["choices"][0]["message"]["content"]  # type: ignore

        return chat
