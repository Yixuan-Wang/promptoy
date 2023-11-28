import os
import dotenv

def test_llamacpp():
    dotenv.load_dotenv()

    from promptoy.interface.message import MessageRole
    from promptoy.provider.llama_cpp import EndpointLlamaCpp

    endpoint = EndpointLlamaCpp()
    messages = [
        (MessageRole.SYSTEM, "You are an assistant good at writing code."),
        (MessageRole.HUMAN, "Write a Hello world program."),
        (MessageRole.ASSISTANT, "In which programming language?"),
        (MessageRole.HUMAN, "Koka. Created by Daan Leijen at Microsoft Research. You can search for examples here: "
                            "https://koka-lang.github.io/koka/doc/book.html"),
    ]
    chat = endpoint.make_chat(
        model=os.environ["PATH_LLAMA_MODEL"],
        gpu_layers=16,
    )
    result = chat(messages)
    print(result)
