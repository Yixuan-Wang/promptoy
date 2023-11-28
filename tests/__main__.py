import dotenv

if __name__ == '__main__':
    dotenv.load_dotenv()

    from promptoy.interface.message import MessageRole
    from promptoy.provider.anthropic import EndpointAnthropic, ModelAnthropic

    endpoint = EndpointAnthropic()
    messages = [
        (MessageRole.SYSTEM, "You are an assistant good at writing code."),
        (MessageRole.HUMAN, "Write a Hello world program."),
        (MessageRole.ASSISTANT, "In which programming language?"),
        (MessageRole.HUMAN, "Koka. Created by Daan Leijen at Microsoft Research. You can search for examples here: "
                            "https://koka-lang.github.io/koka/doc/book.html"),
    ]
    chat = endpoint.make_chat(
        model=ModelAnthropic.CLAUDE,
    )
    result = chat(messages)
    print(result)