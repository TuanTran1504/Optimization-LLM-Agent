import operator
from typing import Dict
from flask import Flask, request, jsonify
import sqlite3
from langchain.schema import Document
from langgraph.graph import END
import json
import getpass
import textwrap
import time
from langgraph.graph.message import add_messages
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import os
from langchain_core.runnables import RunnableConfig
import asyncio
from langgraph.graph import StateGraph
import ollama
import pandas as pd
from solver import solver
from agent import evaluate, briefing, route_question,  node_extraction, greeting, verification
from state import *
from flask_cors import CORS
from flask import Response, stream_with_context, send_file
app = Flask(__name__)
CORS(app)
class Query(BaseModel):
    message: str
    thread_id: str = "1"  # allow multi-user conversation threads

workflow = StateGraph(AgentState)
workflow.add_node("greeting", greeting)
workflow.add_node("verification", verification)
workflow.add_node("route_question", route_question)
workflow.add_node("briefing", briefing)
workflow.add_node("node_extraction", node_extraction)
workflow.add_node("evaluate", evaluate)
workflow.add_node("solver", solver)

workflow.add_edge(START,"greeting")
workflow.add_edge("node_extraction", "briefing")
workflow.add_edge("briefing", "evaluate")
workflow.add_edge("evaluate", "solver")
workflow.add_edge("solver", END)
graph = workflow.compile()

conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
checkpointer = SqliteSaver(conn)
scope = workflow.compile(checkpointer=checkpointer)


@app.route("/threads/<thread_id>/status.json")
def get_thread_status(thread_id):
    """Serve the current node status file inside the thread directory."""
    status_path = os.path.join("threads", thread_id, "status.json")
    if os.path.exists(status_path):
        return send_file(status_path, mimetype="application/json")
    return jsonify({"error": "Status not found"}), 404

@app.route("/chat", methods=["POST"])
def chat_api():
    data = request.get_json()
    message = data.get("message", "")
    thread_id = data.get("thread_id", "1")

    # Handle CSV input
    if message.endswith(".csv"):
        with open(message, "r", encoding="utf-8") as f:
            message = f.read()

    response = "Processing..."
    current_node = "starting"
    constraints, nodes, matrix = {}, [], {}

    # 🧩 Prepare thread directory early
    thread_dir = os.path.join("threads", thread_id)
    os.makedirs(thread_dir, exist_ok=True)

    # --- Stream through the LangGraph workflow ---
    for event in scope.stream(
        {"messages": [HumanMessage(content=message)]},
        config={"configurable": {"thread_id": thread_id}},
    ):
        if not event:
            continue

        # Each event is a dict keyed by node name
        for node_name, payload in event.items():
            if payload is None:
                continue

            # ---- Extract message safely ----
            messages = payload.get("messages") if isinstance(payload, dict) else None
            if messages and isinstance(messages, list):
                last_msg = messages[-1]
                if hasattr(last_msg, "content"):
                    response = last_msg.content

            # ---- Extract state-like data safely ----
            if isinstance(payload, dict):
                # Only update if the field exists and is non-empty
                if "constraints" in payload and payload["constraints"]:
                    constraints = payload["constraints"]
                if "nodes" in payload and payload["nodes"]:
                    nodes = payload["nodes"]
                if "distance_matrix" in payload and payload["distance_matrix"]:
                    matrix = payload["distance_matrix"]

                current_node = payload.get("current_node", node_name)


            # ✅ NEW: write live node status for polling
            try:
                with open(os.path.join(thread_dir, "status.json"), "w", encoding="utf-8") as f:
                    json.dump({"current_node": current_node}, f, indent=2)
            except Exception as e:
                print(f"⚠️ Error writing status.json: {e}")

    # ✅ Save final thread data
    try:
        with open(os.path.join(thread_dir, "constraints.json"), "w", encoding="utf-8") as f:
            json.dump(constraints, f, indent=2)
        with open(os.path.join(thread_dir, "nodes.json"), "w", encoding="utf-8") as f:
            json.dump(nodes, f, indent=2)
        with open(os.path.join(thread_dir, "matrix.json"), "w", encoding="utf-8") as f:
            json.dump(matrix, f, indent=2)
    except Exception as e:
        print(f"⚠️ Error saving thread data: {e}")

    # --- Return final response ---
    return jsonify({
        "response": response,
        "constraints": constraints,
        "current_node": current_node
    })

@app.route("/threads", methods=["GET"])
def get_threads():
    cursor = None
    try:
        cursor = conn.cursor()
        # Fetch unique thread_ids
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints WHERE checkpoint_ns = ''")
        thread_ids = [{"thread_id": row[0]} for row in cursor.fetchall()]
        return jsonify(thread_ids)
    except Exception as e:
        print(f"Error fetching threads: {e}")
        return jsonify({"error": "Failed to fetch threads"}), 500
    finally:
        if cursor:
            cursor.close()
if __name__ == "__main__":
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host="0.0.0.0", port=8000)

