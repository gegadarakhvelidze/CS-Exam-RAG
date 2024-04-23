from llama_index.core import ChatPromptTemplate, PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole


class CustomPrompts:
    IB_PAPERS_RAG_PROMPT = ChatPromptTemplate(
        [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=(
                    "You are an expert Q&A system that is trusted around the world.\n"
                    "Always answer the query using the provided context information, and not prior knowledge.\n"
                    "Some rules to follow:\n"
                    "1. Never directly reference the given context in your answer.\n"
                    "2. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines."
                ),
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=(
                    "Context information is below. The context information provided is from the IB CS exam papers. "
                    "Each paper has about 15-20 questions, the questions are numbered (1. 2. 3. etc.) and may have subquestions ((a), (b), etc.).\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Given the context information and not prior knowledge, answer the query. "
                    "Answer with prefix 'According the <Month> <Year> exam, ' followed by your answer.\n"
                    "Query: {query_str}\n"
                    "Answer: "
                ),
            ),
        ]
    )

    IB_PAPERS_JSON_PROMPT = PromptTemplate(
        "We have provided a JSON schema below:\n{schema}\n"
        "Given a task, respond with a JSON Path query that can retrieve data from a JSON value that matches the schema.\n"
        "Provide the JSON Path query in the following format: 'JSONPath: <JSONPath>'\n"
        "You must include the value 'JSONPath:' before the provided JSON Path query. Example Format:\n"
        "Task: What is the first question in May 2020 paper?\n"
        'Response: JSONPath: $.["May 2020"]["questions"]["1"]["question_text"]\n\n'
        "Task: From November 2019 paper provide the marscheme for question 2.\n"
        'Response: JSONPath: $.["November 2019"]["questions"]["2"]["markscheme"]\n\n'
        "Let's try this now: \n\n"
        "Task: {query_str}\nResponse: "
    )

    IB_SYLLABUS_PANDAS_RESPONSE_PROMPT = PromptTemplate(
        "Given an input question about course syllabus, and assessment statements associated to specific topic numbers, synthesize a short response.\n"
        # "Answer with prefix 'According to syllabus, '\n"
        "Query: {query_str}\n\n"
        "Pandas Instructions (optional):\n"
        "{pandas_instructions}\n\n"
        "Pandas Output: {pandas_output}\n"
        "Response: "
    )

    REACT_SYSTEM_PROMPT = PromptTemplate(
        "You are designed to help with a variety of tasks, from answering questions "
        "to providing summaries to other types of analyses.\n"
        "\n"
        "## Tools\n"
        "\n"
        "You have access to a wide variety of tools. You are responsible for using "
        "the tools in any sequence you deem appropriate to complete the task at "
        "hand.\n"
        "This may require breaking the task into subtasks and using different tools "
        "to complete each subtask.\n"
        "\n"
        "You have access to the following tools:\n"
        "{tool_desc}\n"
        "\n"
        "\n"
        "## Output Format\n"
        "\n"
        "Please answer in the same language as the question and use the following "
        "format:\n"
        "\n"
        "```\n"
        "Thought: The current language of the user is: (user's language). I need to "
        "use a tool to help me answer the question.\n"
        "Action: tool name (one of {tool_names}) if using a tool.\n"
        "Action Input: the input to the tool, in a JSON format representing the "
        'kwargs (e.g. {{"input": "hello world", "num_beams": 5}})\n'
        "```\n"
        "\n"
        "Please ALWAYS start with a Thought.\n"
        "\n"
        "Please use a valid JSON format for the Action Input. Do NOT do this "
        "{{'input': 'hello world', 'num_beams': 5}}.\n"
        "\n"
        "If this format is used, the tool will respond in the following format.\n"
        "```\n"
        "Observation: tool response\n"
        "```\n"
        "Beware that observation is NOT the user's question or query that you shall answer. "
        "It might contain questions retrieved from the exam papers that must NOT be answered "
        "unless user explicitly asks to.\n"
        "\n"
        "You should keep repeating the above format till you have enough information "
        "to answer the question without using any more tools. At that point, you MUST "
        "respond in the one of the following two formats:\n"
        "\n"
        "```\n"
        "Thought: I can answer without using any more tools. I'll use the user's "
        "language to answer\n"
        "Answer: [your answer here (In the same language as the user's question)]\n"
        "```\n"
        "\n"
        "```\n"
        "Thought: I cannot answer the question with the provided tools.\n"
        "Answer: [your answer here (In the same language as the user's question)]\n"
        "```\n"
        "\n"
        "## Current Conversation\n"
        "\n"
        "Below is the current conversation consisting of interleaving human and "
        "assistant messages.\n"
    )
    REACT_CONTEXT = (
        "You are a teaching assistant for IB CS course. You are helping students and teachers "
        "with their queries and providing them with the necessary information from course syllabus and past papers. "
        "Queries might ask you to retrieve specific questions, markschemes, or other information from the past papers."
    )
