from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
import uuid
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool
from langchain.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI(title="DeepSearch AI Agent API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
MONGODB_URI = os.getenv("MONGODB_URI")
client = AsyncIOMotorClient(MONGODB_URI)
db = client.deepsearch_db
conversations_collection = db.conversations
search_results_collection = db.search_results

# Models
class Message(BaseModel):
    role: str
    content: str

class ConversationRequest(BaseModel):
    message: str
    history: Optional[List[Message]] = []

class SearchResult(BaseModel):
    title: str
    snippet: str
    url: str
    relevance: float

class AgentResponse(BaseModel):
    response: str
    search_results: List[SearchResult]
    conversation_id: str

# Initialize LangChain components
def get_agent():
    # Initialize the language model
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Define search tool
    search_tool = TavilySearchResults(
        api_key=os.getenv("TAVILY_API_KEY", "your-tavily-api-key"),
        max_results=5
    )
    
    tools = [
        Tool(
            name="web_search",
            func=search_tool.invoke,
            description="Search the web for information on a given topic. Use this tool when you need to find up-to-date or specific information."
        )
    ]
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an advanced DeepSearch AI agent. Your goal is to provide comprehensive, accurate answers by searching and analyzing information from multiple sources.

When responding:
1. Always use the search tool to find relevant information before answering
2. Cite your sources clearly
3. Synthesize information from multiple sources when possible
4. Be honest about limitations in the available information
5. Provide nuanced answers that consider different perspectives

Your responses should be well-structured, informative, and helpful."""),
        ("human", "{input}"),
    ])
    
    # Create the agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,
    )
    
    return agent_executor

# Routes
@app.get("/")
async def root():
    return {"message": "DeepSearch AI Agent API is running"}

@app.post("/api/agent", response_model=AgentResponse)
async def process_agent_query(request: ConversationRequest):
    try:
        # Format conversation history for the agent
        history = []
        for msg in request.history:
            if msg.role == "user":
                history.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                history.append(AIMessage(content=msg.content))
        
        # Get the agent
        agent_executor = get_agent()
        
        # Execute the agent with the user's query
        agent_response = agent_executor.invoke({
            "input": request.message,
            "chat_history": history
        })
        
        # Extract search results from the agent's intermediate steps
        search_results = []
        for step in agent_response.get("intermediate_steps", []):
            if step[0].tool == "web_search" and isinstance(step[1], list):
                for result in step[1]:
                    search_results.append(
                        SearchResult(
                            title=result.get("title", ""),
                            snippet=result.get("content", ""),
                            url=result.get("url", ""),
                            relevance=float(result.get("score", 0.5))
                        )
                    )
        
        # Generate a conversation ID
        conversation_id = str(uuid.uuid4())
        
        # Store conversation in MongoDB
        await conversations_collection.insert_one({
            "conversation_id": conversation_id,
            "user_message": request.message,
            "agent_response": agent_response["output"],
            "timestamp": datetime.utcnow(),
            "history": [{"role": msg.role, "content": msg.content} for msg in request.history]
        })
        
        # Store search results in MongoDB
        if search_results:
            await search_results_collection.insert_one({
                "conversation_id": conversation_id,
                "results": [result.dict() for result in search_results],
                "timestamp": datetime.utcnow()
            })
        
        return AgentResponse(
            response=agent_response["output"],
            search_results=search_results,
            conversation_id=conversation_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

