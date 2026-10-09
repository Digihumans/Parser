from pathlib import Path
import os
import re
import uuid
import asyncio
from dotenv import load_dotenv

class MarkdownSplitter:
    def __init__(self, max_chunk_tokens:int = 800, min_chunk_tokens:int = 300):
        load_dotenv()

        self.provider=os.getenv("LLM_PROVIDER")

        self.max_chunk_tokens=max_chunk_tokens
        self.min_chunk_tokens=min_chunk_tokens

        tokenizers={
            "aws":os.getenv("AWS_TOKENIZER"),
            "azure":os.getenv("AZURE_TOKENIZER"),
            "ollama":os.getenv("OLLAMA_TOKENIZER")
        }
        if self.provider in ["aws", "azure", "ollama"]:
            tokenizer_name=tokenizers.get(self.provider)
        else:
            raise ValueError(f"'{self.provider}' not supported.")

        match self.provider:
            case "aws":
                ...

            case "azure":
                import tiktoken
                self.tokenizer=tiktoken.get_encoding(tokenizer_name)

            case "ollama":
                from transformers import AutoTokenizer
                tokenizer_path = str(Path(__file__).parent / "tokenizers" / tokenizer_name)
                if os.path.exists(tokenizer_path):
                    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                    self.tokenizer.save_pretrained(tokenizer_path)
    
    async def count_tokens(self, text:str):
        match self.provider:
            case "aws":
                ...
            case "azure":
                return len(self.tokenizer.encode(text))
            case "ollama":
                return len(self.tokenizer.encode(text, add_special_tokens=False))
    
    async def split(self, doc_id:str, source:str, markdown:str)->list[dict]:
        filename=os.path.basename(source)

        # Phase 1-> split by headings
        heading_chunks=await self._split_by_headings(doc_id, filename, markdown)
        # print(heading_chunks)

        # Phase 2-> sub-split by type
        typed_chunks=await self._split_by_type(heading_chunks)
        # print(typed_chunks)

        # Phase 3-> token-aware overflow splitting
        flat_chunks=await self._split_oversized_chunks(typed_chunks)
        print(flat_chunks)

    # Phase 1-> split by headings
    async def _split_by_headings(self, doc_id:str, filename:str, markdown:str)->list[dict]:
        chunks=[]

        current=await self._new_chunk(doc_id, filename, None, None)

        for line in markdown.splitlines():
            stripped=line.strip()

            if stripped.startswith("#") and not stripped.startswith("#[["):
                if current["content"].strip():
                    chunks.append(current)
                
                level=len(stripped)-len(stripped.lstrip("#"))
                current=await self._new_chunk(doc_id, filename, title=stripped, level=level)
            else:
                current["content"]+=line+"\n"
        else:
            if current["content"].strip():
                chunks.append(current)
        return chunks
    
    async def _new_chunk(self, doc_id: str, source: str, title, level) -> dict:
        return {
            "doc_id": doc_id,
            "source": source,
            "chunk_id": uuid.uuid4().hex,
            "title": title,
            "level": level,
            "content": "",
            "chunk_type": "text",
            "parent_chunk_id": None,
            "parent_titles": {},
            "prev_chunk_id": None,
            "next_chunk_id": None,
            "sibling_chunk_ids": [],
            "embedding_text": "",
            "token_count": 0,
            "kb_ids": None,
            "metadata": {},
        }

    # Phase 2-> sub-split by type
    async def _split_by_type(self, heading_chunks:list[dict])->list[dict]:
        all_chunks=[]

        text=""
        code=""
        table=""

        is_code=False
        is_table=False


        for heading_chunk in heading_chunks:
            for line in heading_chunk["content"].splitlines():
                if line.strip().startswith("```") or line.strip().endswith("```"):
                    if not is_code:
                        if text.strip():
                            all_chunks.append(await self._make_sub(heading_chunk, "text", text))
                            text=""
                        if table.strip():
                            all_chunks.append(await self._make_sub(heading_chunk, "table", table))
                            table=""
                            is_table=False
                        is_code=True
                        code+=line+"\n"
                    else:
                        is_code=False
                        code+=line+"\n"
                        all_chunks.append(await self._make_sub(heading_chunk, "code", code))
                        continue
                
                if is_code:
                    code+=line+"\n"
                    continue

                if line.strip().startswith("|"):
                    is_table=True
                    table+=line+"\n"
                    continue

                text+=line+"\n"
            else:
                if code.strip():
                    code_chunk=await self._make_sub(heading_chunk, "code", code)
                    all_chunks.append(code_chunk)
                if table.strip():
                    table_chunk=await self._make_sub(heading_chunk, "table", table)
                    all_chunks.append(table_chunk)
                if text.strip():
                    text_chunk=await self._make_sub(heading_chunk, "text", text)
                    all_chunks.append(text_chunk)
        return all_chunks

    async def _make_sub(self, heading_chunk:dict, chunk_type:str, content:str)->dict:
        return {
            "doc_id": heading_chunk["doc_id"],
            "source": heading_chunk["source"],
            "chunk_id": "",
            "title": heading_chunk["title"],
            "level": heading_chunk["level"],
            "content": content,
            "chunk_type": chunk_type,
            "parent_chunk_id": None,
            "parent_titles": {},
            "prev_chunk_id": None,
            "next_chunk_id": None,
            "sibling_chunk_ids": [],
            "embedding_text": "",
            "token_count": 0,
            "kb_ids": None,
            "metadata": {},
        }
                    
    # Phase 3-> token-aware overflow splitting
    async def _split_oversized_chunks(self, typed_chunks:list[dict]):
        new_chunks=[]

        async def hierarchy_prefix(chunk:dict):
            if chunk.get("title"):
                return chunk["title"].lstrip("#").strip()
            return ""

        for chunk in typed_chunks:
            # Need to build a temp hierarchy prefix for token counting
            prefix=await hierarchy_prefix(chunk)
            embedding_text=prefix+"\n"+chunk.get("content")
            token_count=await self.count_tokens(embedding_text)

            if token_count<=self.max_chunk_tokens or chunk["chunk_type"]=="code":
                chunk["token_count"]=token_count
                new_chunks.append(chunk)
                continue

            if chunk["chunk_type"]=="table":
                # Table splitting logic
                ...
            else:
                # Text splitting logic
                ...

        return new_chunks
                
async def main():
    markdown_splitter=MarkdownSplitter()
    # print(await markdown_splitter.count_tokens("hi, how are you?"))
    await markdown_splitter.split("123", "a23.pdf", "# TOpic 1\nThis is a topic\n```def start():\n\tprint('Hello')\n\tprint('U')```\n|------|\n|name|")

if __name__=="__main__":
    asyncio.run(main())
                