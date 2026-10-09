# from ollama import Client
from dotenv import load_dotenv
import os
import asyncio
from pathlib import Path

class AgenticChunkerError(Exception):
    ...

class Chunker:
    def __init__(self, provider:str="azure"):
        load_dotenv()
        self.provider=provider
        self.tools=[
            {
                "type":"function",
                "function":{
                    "name":"extract_chunks",
                    "description":("Split the provided markdown into semantically coherent chunks. "
                    "Boundaries are character offsets into window text. The union of "
                    "chunks must cover all non-whitespace content of the window with no overlaps."),
                    "parameters":{
                        "type":"object",
                        "properties":{
                            "chunks": {
                                "type":"array",
                                "description":"Ordered list of chunks covering the window. Each chunk's [start:end] is a character offset range into the window text",
                                "items":{
                                    "type":"object",
                                    "properties": {
                                        "start": {
                                            "type":"integer",
                                            "description":"Character offset within the window. Must satisfy 0 <= start < end <= length(window_text)"
                                        },
                                        "end":{
                                            "type":"integer",
                                            "description":"Character offset within the window. Must satisfy start < end <= length(window_text)"
                                        },
                                        "title":{
                                            "type":"string",
                                            "description":"The title of the window text. Do not share the titles unless the chunks are siblings in the same heading section"
                                        }
                                    },
                                    "required":["start", "end", "title"]
                                }
                            }
                        },
                        "required":["chunks"]
                    }
                }
            }
        ]
        
        match provider:
            case "azure":
                from openai import AsyncAzureOpenAI
                AZURE_ENDPOINT=os.getenv("AZURE_ENDPOINT")
                API_VERSION=os.getenv("AZURE_LLM_API_VERSION")
                API_KEY=os.getenv("AZURE_API_KEY")
                MODEL="gpt-5.4-mini"

                self.azure_async_client=AsyncAzureOpenAI(
                    azure_endpoint=AZURE_ENDPOINT,
                    api_version=API_VERSION,
                    api_key=API_KEY
                )
                self.model=MODEL
            
            case "ollama":
                from ollama import AsyncClient
                self.ollama_async_client=AsyncClient()
            
            case "aws":
                import boto3
                self.aws_client=boto3.client("bedrock-runtime")
            
            case _:
                raise ValueError(f"Provider '{provider}' not supported")
    
    async def chunk(self, chunks:list):
        messages=[
            {"role":"system", "content":"You are an agentic chunker, who chunks the text into semantically coherent chunks"},
            {"role":"user", "content":f"Chunks are '[{chunks}]'"}
        ]
        try:
            match self.provider:
                case "azure":
                    response=await self.azure_async_client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=self.tools,
                        # tool_choice={"type":"function","function":{"name":"extract_chunk_data"}},
                        stream=False
                    )

                    if response.choices[0].message.tool_calls:
                        function_name=response.choices[0].message.tool_calls[0].function
                        print(function_name)
                    else:
                        print(response)
                    # print(response.choices[0].message)
                case "aws":
                    def construct_aws_message(messages):
                        messages=[{"role":message.get("role"), "content":[{"text":message.get("content")}]} for message in messages[1:]]
                        return messages
                    
                    def invoke_llm_sync():
                        response=self.aws_client.converse(
                            modelId="openai.gpt-oss-20b-1:0",
                            system=[
                                {"text":"You are an agentic chunker, who chunks the text into semantically coherent chunks"}
                            ],
                            messages=construct_aws_message(messages),
                            
                        )
                        print(response)
                    await asyncio.to_thread(invoke_llm_sync)


                case "ollama":
                    ...
                
        except Exception as e:
            raise AgenticChunkerError(f"Could not make chunks: {str(e)}")

if __name__=="__main__":
    chunker=Chunker("aws")
    with open("C:/Users/arshs/Desktop/Projects/RAG"+"/files/resume_1.md", "r") as f:
        markdown=f.read()
    asyncio.run(chunker.chunk([markdown]))
