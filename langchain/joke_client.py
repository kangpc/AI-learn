from langserve import RemoteRunnable

joke_chain = RemoteRunnable("http://localhost:8088/joke/")

resp = joke_chain.invoke({"topic": "小明"})
print(resp)