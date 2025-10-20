# Optimization LLM: 

🔍 Overview

This project explores how Generative AI and optimization can be combined to make complex problem-solving more accessible. The system is designed to let users define Travelling Salesman Problem (TSP) scenarios through natural language conversation, which are then interpreted, formalized, and solved automatically. Built with LangGraph, the framework integrates conversational input, semantic parsing, and optimization tools into a coherent, intelligent agent system.

🧩 Core Components

Conversational Problem Specification – Users describe TSP scenarios via dialogue; the system extracts entities, routes, and constraints.

Input Parsing and Formalization – The input is transformed into structured parameters suitable for computational processing.

Optimization and Solution Delivery – The system integrates high-performance solvers (e.g., OR-Tools, Concorde, LKH) to compute efficient routes and explain the results clearly.



## ⚙️ Setup


1. Clone the repo:
   ```bash
   git clone https://github.com/TuanTran1504/Optimization-LLM-Agent
   cd Optimization-LLM-Agent
   ```

2. Start the env:
   ```bash
   conda create --name optagent
   conda activate optagent
   pip install -r requirements.txt
   ```
3. Start Backend:
   ```bash
   python main.py
   ```
4. Access the web UI:
   ```
   search.html
   ```
5. Keys
Open AI Key and Google Map API Key is needed to run this \
Put them in a .env file in the root and in the extraction_tools folders
or else:
You will have to provide the distance matrix or sufficient measurements for the agent to solve the problem


## ⚙️ Test Data

The problem descriptions in the data folder can be used to test the agent and its ability to extract constraints from user input 

<p align="center">
  <img src="data/images/images.jpeg" alt="Example Image" width="1000"/>
</p>

## 📄 License

MIT License © 2025 Dinh Tuan Tran
