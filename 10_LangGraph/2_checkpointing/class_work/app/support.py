# from .graph import graph
from dotenv import load_dotenv
from langgraph.checkpoint.mongodb import MongoDBSaver
from .graph import create_chat_graph
import json
from langgraph.types import Command

load_dotenv()


MONGODB_URI = "mongodb://localhost:27017"
config = {"configurable": {"thread_id": "3"}}


def init():
    with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
        graph_with_mongo = create_chat_graph(checkpointer=checkpointer)

        state = graph_with_mongo.get_state(config=config)
        # print("state: ", state.values["messages"])

        # for message in state.values["messages"]:
        #     message.pretty_print()

        last_message = state.values["messages"][
            -1
        ]  # last message because to check it is tool call or not
        tool_calls = last_message.additional_kwargs.get("tool_calls", [])

        user_query = None

        for call in tool_calls:
            if call.get("function", {}).get("name") == "human_assistance_tool":
                args = call["function"].get("arguments", "{}")
                try:
                    args_dict = json.loads(args)
                    user_query = args_dict.get("query")
                except json.JSONDecodeError:
                    print("Failed to decode function arguments.")

        print("User is trying to Ask: ", user_query)
        ans = input("Resolution >> ")

        resume_command = Command(
            resume={"data": ans}
        )  # resuming the graph because it was interupted because ai did'nt know the answer
        # graph_with_mongo.stream(resume_command, config=config, stream_mode="values")
        for event in graph_with_mongo.stream(
            resume_command, config=config, stream_mode="values"
        ):
            if "messages" in event:
                event["messages"][-1].pretty_print()


init()
