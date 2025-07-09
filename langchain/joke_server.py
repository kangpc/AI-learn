#!/usr/bin/env python

"""
langserve 用于将Chain或者Runnable部署成一个REST API服务
"""

from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langserve import add_routes

from dotenv import load_dotenv
load_dotenv()
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)

llm = ChatOpenAI(
    model="Qwen/Qwen2.5-7B-Instruct", 
    temperature=0,
    base_url="https://api.siliconflow.cn",
    api_key="sk-***"
)

prompt = ChatPromptTemplate.from_template("讲一个关于{topic}的笑话")
add_routes(
    app,
    prompt | llm,
    path="/joke",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8088)