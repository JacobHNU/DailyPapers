# 实践项目3：构建RAG记忆系统

> 新手友好教程 · 检索增强生成 · 向量存储 · 记忆管理
> 预计时间：3-4小时
> 难度：⭐⭐ 初级-中级

---

## 项目概述

```
项目目标
═══════════════════════════════════════════════════════════════════

构建一个RAG（检索增强生成）记忆系统，能够：
1. 存储文档和知识
2. 将文档分割成块
3. 生成嵌入向量
4. 在向量数据库中索引
5. 根据查询检索相关文档
6. 使用检索到的文档增强LLM生成

核心思想：给LLM提供外部记忆
- LLM本身没有长期记忆
- RAG提供动态、可更新的外部知识
- 检索相关信息增强生成质量

示例场景：
用户："公司的请假政策是什么？"
系统：检索公司手册中的请假政策
LLM：基于检索到的政策生成答案

═══════════════════════════════════════════════════════════════════
```

---

## 第一步：理解RAG架构

### 1.1 RAG核心组件

```
RAG架构
═══════════════════════════════════════════════════════════════════

离线索引阶段：
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  文档加载    │ →  │   分块      │ →  │   嵌入      │
│             │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
                                            │
                                            ↓
                                     ┌─────────────┐
                                     │  向量数据库  │
                                     └─────────────┘

在线查询阶段：
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  用户查询   │ →  │  查询嵌入   │ →  │  相似度检索  │
└─────────────┘    └─────────────┘    └─────────────┘
                                            │
                                            ↓
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  最终答案   │ ←  │   LLM生成   │ ←  │  上下文增强  │
└─────────────┘    └─────────────┘    └─────────────┘

═══════════════════════════════════════════════════════════════════
```

### 1.2 关键概念

```
关键概念
═══════════════════════════════════════════════════════════════════

1. 文档分块（Chunking）
   - 将长文档分割成小块
   - 每块通常256-512令牌
   - 保持语义连贯性

2. 嵌入（Embedding）
   - 将文本转换为向量
   - 相似文本 → 相似向量
   - 使用预训练模型

3. 向量数据库
   - 存储和检索向量
   - 支持相似度搜索
   - 如ChromaDB、FAISS、Pinecone

4. 检索（Retrieval）
   - 根据查询找到相关文档
   - 使用余弦相似度
   - 返回top-k结果

5. 增强生成（Augmented Generation）
   - 将检索到的文档注入提示
   - LLM基于上下文生成答案
   - 减少幻觉，提高准确性

═══════════════════════════════════════════════════════════════════
```

---

## 第二步：环境准备

### 2.1 安装依赖

```bash
# 创建虚拟环境
python -m venv rag_memory_env
source rag_memory_env/bin/activate  # Linux/Mac
# 或
rag_memory_env\Scripts\activate  # Windows

# 安装依赖
pip install openai langchain langchain-openai chromadb tiktoken python-dotenv
```

### 2.2 项目结构

```
rag_memory_project/
├── .env                    # 环境变量
├── requirements.txt        # 依赖列表
├── README.md              # 项目说明
│
├── memory/                # 记忆系统
│   ├── __init__.py
│   ├── document_loader.py # 文档加载器
│   ├── text_splitter.py   # 文本分割器
│   ├── embeddings.py      # 嵌入模型
│   ├── vector_store.py    # 向量存储
│   └── rag_memory.py      # RAG记忆核心
│
├── utils/                 # 工具函数
│   ├── __init__.py
│   └── helpers.py
│
├── examples/              # 示例
│   ├── basic_example.py
│   ├── document_example.py
│   └── conversation_example.py
│
└── main.py               # 主入口
```

---

## 第三步：实现核心组件

### 3.1 文档加载器

```python
# memory/document_loader.py
from typing import List, Dict, Any
from dataclasses import dataclass
import os

@dataclass
class Document:
    """文档数据类"""
    page_content: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DocumentLoader:
    """
    文档加载器
    支持多种格式的文档加载
    """
    
    @staticmethod
    def load_text(text: str, metadata: Dict = None) -> Document:
        """
        从文本加载文档
        
        参数:
            text: 文本内容
            metadata: 元数据
            
        返回:
            Document对象
        """
        return Document(
            page_content=text,
            metadata=metadata or {}
        )
    
    @staticmethod
    def load_from_file(file_path: str) -> List[Document]:
        """
        从文件加载文档
        
        参数:
            file_path: 文件路径
            
        返回:
            Document列表
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        documents = []
        
        # 根据文件扩展名选择加载方式
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append(Document(
                    page_content=content,
                    metadata={"source": file_path}
                ))
        
        elif ext == '.md':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append(Document(
                    page_content=content,
                    metadata={"source": file_path, "type": "markdown"}
                ))
        
        elif ext == '.csv':
            import csv
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    content = ' | '.join(f"{k}: {v}" for k, v in row.items())
                    documents.append(Document(
                        page_content=content,
                        metadata={"source": file_path, "row": i}
                    ))
        
        else:
            # 默认作为文本处理
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append(Document(
                    page_content=content,
                    metadata={"source": file_path}
                ))
        
        return documents
    
    @staticmethod
    def load_from_directory(directory: str, extensions: List[str] = None) -> List[Document]:
        """
        从目录加载所有文档
        
        参数:
            directory: 目录路径
            extensions: 文件扩展名过滤
            
        返回:
            Document列表
        """
        if extensions is None:
            extensions = ['.txt', '.md', '.csv']
        
        documents = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    try:
                        docs = DocumentLoader.load_from_file(file_path)
                        documents.extend(docs)
                    except Exception as e:
                        print(f"加载文件失败 {file_path}: {e}")
        
        return documents
```

### 3.2 文本分割器

```python
# memory/text_splitter.py
from typing import List
from memory.document_loader import Document


class TextSplitter:
    """
    文本分割器
    将长文本分割成小块
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: List[str] = None
    ):
        """
        初始化分割器
        
        参数:
            chunk_size: 每块的目标大小（字符数）
            chunk_overlap: 块之间的重叠字符数
            separators: 分隔符列表
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " "]
    
    def split_text(self, text: str) -> List[str]:
        """
        分割文本
        
        参数:
            text: 要分割的文本
            
        返回:
            文本块列表
        """
        # 如果文本足够短，直接返回
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        
        # 尝试使用分隔符分割
        for separator in self.separators:
            if separator in text:
                splits = text.split(separator)
                
                current_chunk = ""
                for split in splits:
                    # 如果当前块加上新分割超过大小限制
                    if len(current_chunk) + len(split) + len(separator) > self.chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        
                        # 开始新块，可能包含重叠
                        if self.chunk_overlap > 0 and chunks:
                            # 取上一块的末尾作为重叠
                            last_chunk = chunks[-1]
                            overlap_text = last_chunk[-self.chunk_overlap:]
                            current_chunk = overlap_text + separator + split
                        else:
                            current_chunk = split
                    else:
                        if current_chunk:
                            current_chunk += separator + split
                        else:
                            current_chunk = split
                
                # 添加最后一块
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                return chunks
        
        # 如果没有找到分隔符，按固定大小分割
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        分割文档列表
        
        参数:
            documents: 文档列表
            
        返回:
            分割后的文档列表
        """
        split_docs = []
        
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            
            for i, chunk in enumerate(chunks):
                # 创建新文档，保留原始元数据
                new_metadata = doc.metadata.copy()
                new_metadata["chunk_index"] = i
                new_metadata["total_chunks"] = len(chunks)
                
                split_docs.append(Document(
                    page_content=chunk,
                    metadata=new_metadata
                ))
        
        return split_docs
```

### 3.3 嵌入模型

```python
# memory/embeddings.py
from typing import List
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class EmbeddingModel:
    """
    嵌入模型
    将文本转换为向量
    """
    
    def __init__(self, model: str = "text-embedding-ada-002"):
        """
        初始化嵌入模型
        
        参数:
            model: 使用的嵌入模型
        """
        self.client = OpenAI()
        self.model = model
    
    def embed_text(self, text: str) -> List[float]:
        """
        生成单个文本的嵌入
        
        参数:
            text: 输入文本
            
        返回:
            嵌入向量
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        
        return response.data[0].embedding
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成文本嵌入
        
        参数:
            texts: 文本列表
            
        返回:
            嵌入向量列表
        """
        # 分批处理（API限制）
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    def get_dimension(self) -> int:
        """
        获取嵌入维度
        
        返回:
            嵌入维度
        """
        # text-embedding-ada-002 的维度是 1536
        if self.model == "text-embedding-ada-002":
            return 1536
        # 可以通过实际调用获取
        test_embedding = self.embed_text("test")
        return len(test_embedding)


class LocalEmbeddingModel:
    """
    本地嵌入模型（不依赖API）
    使用sentence-transformers
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        初始化本地嵌入模型
        
        参数:
            model_name: 模型名称
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError("请安装sentence-transformers: pip install sentence-transformers")
    
    def embed_text(self, text: str) -> List[float]:
        """生成单个文本的嵌入"""
        return self.model.encode(text).tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量生成文本嵌入"""
        return self.model.encode(texts).tolist()
    
    def get_dimension(self) -> int:
        """获取嵌入维度"""
        return self.dimension
```

### 3.4 向量存储

```python
# memory/vector_store.py
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
from memory.document_loader import Document


class VectorStore:
    """
    向量存储
    使用ChromaDB存储和检索向量
    """
    
    def __init__(
        self,
        collection_name: str = "default",
        persist_directory: str = "./chroma_db",
        embedding_function=None
    ):
        """
        初始化向量存储
        
        参数:
            collection_name: 集合名称
            persist_directory: 持久化目录
            embedding_function: 嵌入函数
        """
        # 创建ChromaDB客户端
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.embedding_function = embedding_function
    
    def add_documents(
        self, 
        documents: List[Document], 
        embeddings: List[List[float]] = None
    ):
        """
        添加文档到向量存储
        
        参数:
            documents: 文档列表
            embeddings: 预计算的嵌入（可选）
        """
        if not documents:
            return
        
        # 准备数据
        ids = [f"doc_{i}" for i in range(len(documents))]
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # 如果没有提供嵌入，使用嵌入函数计算
        if embeddings is None and self.embedding_function:
            embeddings = self.embedding_function(texts)
        
        # 添加到集合
        if embeddings:
            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
        else:
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
    
    def search(
        self, 
        query: str, 
        k: int = 5,
        query_embedding: List[float] = None
    ) -> List[Tuple[Document, float]]:
        """
        搜索相似文档
        
        参数:
            query: 查询文本
            k: 返回结果数量
            query_embedding: 预计算的查询嵌入
            
        返回:
            (文档, 相似度分数) 列表
        """
        # 如果没有提供嵌入，使用嵌入函数计算
        if query_embedding is None and self.embedding_function:
            query_embedding = self.embedding_function([query])[0]
        
        # 执行搜索
        if query_embedding:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
        else:
            results = self.collection.query(
                query_texts=[query],
                n_results=k
            )
        
        # 转换结果
        search_results = []
        
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                doc = Document(
                    page_content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i] if results['metadatas'] else {}
                )
                score = 1 - results['distances'][0][i] if results['distances'] else 0
                search_results.append((doc, score))
        
        return search_results
    
    def delete_all(self):
        """删除所有文档"""
        # 获取所有文档ID
        results = self.collection.get()
        if results['ids']:
            self.collection.delete(ids=results['ids'])
    
    def get_count(self) -> int:
        """获取文档数量"""
        return self.collection.count()
```

### 3.5 RAG记忆核心

```python
# memory/rag_memory.py
from typing import List, Optional
from memory.document_loader import DocumentLoader, Document
from memory.text_splitter import TextSplitter
from memory.embeddings import EmbeddingModel
from memory.vector_store import VectorStore


class RAGMemory:
    """
    RAG记忆系统
    整合文档加载、分割、嵌入和检索
    """
    
    def __init__(
        self,
        embedding_model: str = "text-embedding-ada-002",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        collection_name: str = "rag_memory",
        persist_directory: str = "./chroma_db"
    ):
        """
        初始化RAG记忆系统
        
        参数:
            embedding_model: 嵌入模型名称
            chunk_size: 文本块大小
            chunk_overlap: 块重叠大小
            collection_name: 向量存储集合名称
            persist_directory: 向量存储持久化目录
        """
        # 初始化组件
        self.loader = DocumentLoader()
        self.splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.embedder = EmbeddingModel(model=embedding_model)
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=self.embedder.embed_texts
        )
        
        print(f"RAG记忆系统初始化完成")
        print(f"  嵌入模型: {embedding_model}")
        print(f"  块大小: {chunk_size}")
        print(f"  集合名称: {collection_name}")
    
    def add_text(self, text: str, metadata: dict = None):
        """
        添加文本到记忆
        
        参数:
            text: 文本内容
            metadata: 元数据
        """
        # 创建文档
        doc = self.loader.load_text(text, metadata)
        
        # 分割文档
        chunks = self.splitter.split_documents([doc])
        
        # 生成嵌入
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embedder.embed_texts(texts)
        
        # 添加到向量存储
        self.vector_store.add_documents(chunks, embeddings)
        
        print(f"添加了 {len(chunks)} 个文本块")
    
    def add_file(self, file_path: str):
        """
        添加文件到记忆
        
        参数:
            file_path: 文件路径
        """
        # 加载文档
        docs = self.loader.load_from_file(file_path)
        
        # 分割文档
        chunks = self.splitter.split_documents(docs)
        
        # 生成嵌入
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embedder.embed_texts(texts)
        
        # 添加到向量存储
        self.vector_store.add_documents(chunks, embeddings)
        
        print(f"从文件 {file_path} 添加了 {len(chunks)} 个文本块")
    
    def add_directory(self, directory: str, extensions: List[str] = None):
        """
        添加目录中的所有文件到记忆
        
        参数:
            directory: 目录路径
            extensions: 文件扩展名过滤
        """
        # 加载所有文档
        docs = self.loader.load_from_directory(directory, extensions)
        
        # 分割文档
        chunks = self.splitter.split_documents(docs)
        
        # 生成嵌入
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embedder.embed_texts(texts)
        
        # 添加到向量存储
        self.vector_store.add_documents(chunks, embeddings)
        
        print(f"从目录 {directory} 添加了 {len(chunks)} 个文本块")
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        搜索相关文档
        
        参数:
            query: 查询文本
            k: 返回结果数量
            
        返回:
            相关文档列表
        """
        # 生成查询嵌入
        query_embedding = self.embedder.embed_text(query)
        
        # 搜索
        results = self.vector_store.search(
            query=query,
            k=k,
            query_embedding=query_embedding
        )
        
        # 提取文档
        documents = [doc for doc, score in results]
        
        return documents
    
    def search_with_scores(self, query: str, k: int = 5) -> List[tuple]:
        """
        搜索相关文档（带分数）
        
        参数:
            query: 查询文本
            k: 返回结果数量
            
        返回:
            (文档, 分数) 列表
        """
        query_embedding = self.embedder.embed_text(query)
        
        return self.vector_store.search(
            query=query,
            k=k,
            query_embedding=query_embedding
        )
    
    def get_context(self, query: str, k: int = 3) -> str:
        """
        获取查询的上下文
        
        参数:
            query: 查询文本
            k: 返回结果数量
            
        返回:
            格式化的上下文字符串
        """
        documents = self.search(query, k)
        
        if not documents:
            return "未找到相关信息。"
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[文档 {i}]")
            context_parts.append(doc.page_content)
            if doc.metadata.get('source'):
                context_parts.append(f"来源: {doc.metadata['source']}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def clear(self):
        """清除所有记忆"""
        self.vector_store.delete_all()
        print("记忆已清除")
    
    def get_stats(self) -> dict:
        """
        获取记忆统计信息
        
        返回:
            统计信息字典
        """
        return {
            "total_documents": self.vector_store.get_count(),
        }
```

---

## 第四步：集成LLM生成

### 4.1 RAG增强的LLM

```python
# memory/rag_llm.py
from typing import List, Optional
from openai import OpenAI
from memory.rag_memory import RAGMemory


class RAGLLM:
    """
    RAG增强的LLM
    使用检索到的文档增强生成
    """
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        memory: RAGMemory = None,
        top_k: int = 3
    ):
        """
        初始化RAG LLM
        
        参数:
            model: 使用的LLM模型
            memory: RAG记忆系统
            top_k: 检索的文档数量
        """
        self.client = OpenAI()
        self.model = model
        self.memory = memory or RAGMemory()
        self.top_k = top_k
    
    def generate(self, query: str, use_context: bool = True) -> str:
        """
        生成回答
        
        参数:
            query: 用户问题
            use_context: 是否使用上下文
            
        返回:
            生成的回答
        """
        if use_context:
            # 检索相关文档
            context = self.memory.get_context(query, k=self.top_k)
            
            # 构建带上下文的提示
            prompt = f"""基于以下上下文信息回答用户问题。如果上下文中没有相关信息，请说明你不确定。

上下文：
{context}

用户问题：{query}

回答："""
        else:
            # 不使用上下文
            prompt = query
        
        # 调用LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def chat(self, messages: List[dict], use_context: bool = True) -> str:
        """
        多轮对话
        
        参数:
            messages: 对话历史
            use_context: 是否使用上下文
            
        返回:
            生成的回答
        """
        # 获取最后一个用户消息
        last_message = messages[-1]['content']
        
        if use_context:
            # 检索相关文档
            context = self.memory.get_context(last_message, k=self.top_k)
            
            # 构建系统消息
            system_message = f"""你是一个有用的助手。基于以下上下文信息回答用户问题。
如果上下文中没有相关信息，请说明你不确定。

上下文：
{context}"""
            
            # 构建完整消息
            full_messages = [{"role": "system", "content": system_message}] + messages
        else:
            full_messages = messages
        
        # 调用LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content
```

---

## 第五步：测试RAG记忆系统

### 5.1 基本示例

```python
# examples/basic_example.py
from memory.rag_memory import RAGMemory
from memory.rag_llm import RAGLLM


def test_basic_rag():
    """测试基本RAG功能"""
    
    print("="*60)
    print("测试基本RAG功能")
    print("="*60)
    
    # 1. 创建RAG记忆系统
    memory = RAGMemory(
        chunk_size=200,
        chunk_overlap=20
    )
    
    # 2. 添加一些知识
    print("\n添加知识...")
    
    knowledge_texts = [
        "Python是一种高级编程语言，由Guido van Rossum于1991年创建。Python的设计哲学强调代码的可读性和简洁性。",
        "Python支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。它具有动态类型系统和自动内存管理。",
        "Python的标准库非常庞大，提供了丰富的模块和函数。常见的库包括NumPy、Pandas、Matplotlib等。",
        "机器学习是人工智能的一个子领域，它使计算机能够从数据中学习，而无需显式编程。常见的机器学习算法包括线性回归、决策树、神经网络等。",
        "深度学习是机器学习的一个子集，使用多层神经网络来学习数据的复杂模式。常见的深度学习框架包括TensorFlow、PyTorch等。",
    ]
    
    for text in knowledge_texts:
        memory.add_text(text)
    
    # 3. 搜索测试
    print("\n搜索测试...")
    
    queries = [
        "Python是什么时候创建的？",
        "Python有哪些编程范式？",
        "什么是机器学习？",
        "深度学习和机器学习有什么关系？",
    ]
    
    for query in queries:
        print(f"\n查询: {query}")
        results = memory.search(query, k=2)
        
        for i, doc in enumerate(results, 1):
            print(f"  结果{i}: {doc.page_content[:100]}...")
    
    # 4. 使用LLM生成回答
    print("\n\nLLM生成测试...")
    
    rag_llm = RAGLLM(memory=memory, top_k=2)
    
    for query in queries:
        print(f"\n问题: {query}")
        answer = rag_llm.generate(query)
        print(f"回答: {answer}")
    
    # 5. 显示统计信息
    print(f"\n\n统计信息: {memory.get_stats()}")


if __name__ == "__main__":
    test_basic_rag()
```

### 5.2 文件加载示例

```python
# examples/document_example.py
from memory.rag_memory import RAGMemory
from memory.rag_llm import RAGLLM


def test_document_loading():
    """测试文档加载功能"""
    
    print("="*60)
    print("测试文档加载功能")
    print("="*60)
    
    # 1. 创建RAG记忆系统
    memory = RAGMemory(chunk_size=300, chunk_overlap=30)
    
    # 2. 创建示例文档
    print("\n创建示例文档...")
    
    # 创建一个示例文本文件
    sample_text = """公司员工手册

第一章：入职指南

1.1 入职流程
新员工入职需要完成以下步骤：
- 提交个人资料和证件
- 签署劳动合同
- 参加入职培训
- 领取办公设备和账号

1.2 试用期
试用期为3个月，期间双方可随时解除合同。
试用期结束后，将进行转正评估。

第二章：考勤制度

2.1 工作时间
标准工作时间为周一至周五，上午9:00至下午6:00。
午休时间为12:00至13:00。

2.2 请假制度
员工请假需提前申请：
- 病假：需提供医院证明
- 事假：需提前3天申请
- 年假：入职满1年后享有5天年假

第三章：薪酬福利

3.1 薪资结构
薪资由基本工资、绩效奖金和补贴组成。
每月15日发放上月工资。

3.2 福利待遇
公司提供以下福利：
- 五险一金
- 商业保险
- 年度体检
- 节日礼品
"""
    
    # 保存为文件
    with open("sample_handbook.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)
    
    # 3. 加载文档
    print("\n加载文档...")
    memory.add_file("sample_handbook.txt")
    
    # 4. 查询测试
    print("\n查询测试...")
    
    queries = [
        "公司的请假制度是什么？",
        "新员工入职需要做什么？",
        "公司的薪资结构是怎样的？",
        "试用期多长时间？",
    ]
    
    rag_llm = RAGLLM(memory=memory, top_k=2)
    
    for query in queries:
        print(f"\n问题: {query}")
        answer = rag_llm.generate(query)
        print(f"回答: {answer}")
    
    # 5. 清理
    import os
    if os.path.exists("sample_handbook.txt"):
        os.remove("sample_handbook.txt")
    
    print(f"\n统计信息: {memory.get_stats()}")


if __name__ == "__main__":
    test_document_loading()
```

### 5.3 对话示例

```python
# examples/conversation_example.py
from memory.rag_memory import RAGMemory
from memory.rag_llm import RAGLLM


def test_conversation():
    """测试多轮对话功能"""
    
    print("="*60)
    print("测试多轮对话功能")
    print("="*60)
    
    # 1. 创建RAG记忆系统
    memory = RAGMemory(chunk_size=200)
    
    # 2. 添加知识
    print("\n添加知识...")
    
    knowledge = [
        "Python是一种解释型语言，代码在运行时逐行解释执行。",
        "Python的包管理器是pip，用于安装和管理第三方库。",
        "虚拟环境（venv）用于隔离不同项目的依赖。",
        "Git是一个分布式版本控制系统，用于跟踪代码变更。",
        "GitHub是一个基于Git的代码托管平台。",
    ]
    
    for text in knowledge:
        memory.add_text(text)
    
    # 3. 创建RAG LLM
    rag_llm = RAGLLM(memory=memory, top_k=2)
    
    # 4. 模拟多轮对话
    print("\n开始对话...")
    
    conversations = [
        "什么是Python？",
        "如何安装Python包？",
        "为什么需要虚拟环境？",
        "Git是什么？",
        "GitHub和Git有什么区别？",
    ]
    
    messages = []
    
    for question in conversations:
        print(f"\n用户: {question}")
        
        # 添加用户消息
        messages.append({"role": "user", "content": question})
        
        # 生成回答
        answer = rag_llm.chat(messages)
        print(f"助手: {answer}")
        
        # 添加助手消息
        messages.append({"role": "assistant", "content": answer})
    
    print(f"\n统计信息: {memory.get_stats()}")


if __name__ == "__main__":
    test_conversation()
```

---

## 第六步：高级功能

### 6.1 持久化存储

```python
# memory/persistent_memory.py
import json
import os
from typing import List
from memory.rag_memory import RAGMemory


class PersistentRAGMemory(RAGMemory):
    """
    持久化RAG记忆系统
    支持保存和加载状态
    """
    
    def save_state(self, filepath: str):
        """
        保存状态
        
        参数:
            filepath: 保存路径
        """
        state = {
            "stats": self.get_stats(),
            "config": {
                "chunk_size": self.splitter.chunk_size,
                "chunk_overlap": self.splitter.chunk_overlap,
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        
        print(f"状态已保存到 {filepath}")
    
    def load_state(self, filepath: str):
        """
        加载状态
        
        参数:
            filepath: 状态文件路径
        """
        if not os.path.exists(filepath):
            print(f"状态文件不存在: {filepath}")
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        print(f"加载状态: {state}")
```

### 6.2 带过滤的搜索

```python
# 在VectorStore类中添加
def search_with_filter(
    self, 
    query: str, 
    k: int = 5,
    filter_dict: dict = None
) -> List[Tuple[Document, float]]:
    """
    带过滤的搜索
    
    参数:
        query: 查询文本
        k: 返回结果数量
        filter_dict: 过滤条件
        
    返回:
        (文档, 分数) 列表
    """
    query_embedding = self.embedding_function([query])[0]
    
    # 构建查询参数
    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": k
    }
    
    # 添加过滤条件
    if filter_dict:
        query_params["where"] = filter_dict
    
    results = self.collection.query(**query_params)
    
    # 转换结果
    search_results = []
    
    if results['documents'] and results['documents'][0]:
        for i in range(len(results['documents'][0])):
            doc = Document(
                page_content=results['documents'][0][i],
                metadata=results['metadatas'][0][i] if results['metadatas'] else {}
            )
            score = 1 - results['distances'][0][i] if results['distances'] else 0
            search_results.append((doc, score))
    
    return search_results
```

### 6.3 文档更新和删除

```python
# 在RAGMemory类中添加
def update_document(self, doc_id: str, new_text: str, metadata: dict = None):
    """
    更新文档
    
    参数:
        doc_id: 文档ID
        new_text: 新文本
        metadata: 新元数据
    """
    # 分割新文本
    doc = Document(page_content=new_text, metadata=metadata or {})
    chunks = self.splitter.split_documents([doc])
    
    # 生成嵌入
    texts = [chunk.page_content for chunk in chunks]
    embeddings = self.embedder.embed_texts(texts)
    
    # 更新向量存储
    # 注意：ChromaDB的更新API可能需要调整
    self.vector_store.collection.update(
        ids=[doc_id],
        documents=[new_text],
        embeddings=embeddings[0] if embeddings else None,
        metadatas=[metadata] if metadata else None
    )
    
    print(f"文档 {doc_id} 已更新")

def delete_document(self, doc_id: str):
    """
    删除文档
    
    参数:
        doc_id: 文档ID
    """
    self.vector_store.collection.delete(ids=[doc_id])
    print(f"文档 {doc_id} 已删除")
```

---

## 第七步：完整主入口

```python
# main.py
from memory.rag_memory import RAGMemory
from memory.rag_llm import RAGLLM
from examples.basic_example import test_basic_rag
from examples.document_example import test_document_loading
from examples.conversation_example import test_conversation


def main():
    print("RAG记忆系统实践项目")
    print("="*60)
    
    # 选择示例
    examples = {
        "1": ("基本RAG功能", test_basic_rag),
        "2": ("文档加载", test_document_loading),
        "3": ("多轮对话", test_conversation),
    }
    
    print("\n可用示例:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    
    choice = input("\n选择示例 (1-3, 或 'all' 运行所有): ").strip()
    
    if choice == 'all':
        for name, test_func in examples.values():
            print(f"\n\n{'='*60}")
            print(f"示例: {name}")
            print('='*60)
            test_func()
    elif choice in examples:
        name, test_func = examples[choice]
        print(f"\n运行示例: {name}")
        test_func()
    else:
        print("无效选择")


if __name__ == "__main__":
    main()
```

---

## 第八步：常见问题和解决方案

```
常见问题
═══════════════════════════════════════════════════════════════════

问题1：嵌入API调用失败
症状：API错误或超时
解决方案：
- 检查API密钥
- 添加重试机制
- 考虑使用本地嵌入模型

问题2：搜索结果不相关
症状：检索到的文档与查询无关
解决方案：
- 调整块大小
- 改进嵌入模型
- 添加更多相关文档

问题3：内存使用过高
症状：大量文档导致内存溢出
解决方案：
- 使用持久化存储
- 分批处理文档
- 优化块大小

问题4：响应时间长
症状：检索和生成都很慢
解决方案：
- 缓存常见查询
- 减少检索数量
- 使用更快的嵌入模型

问题5：答案质量差
症状：LLM生成的答案不准确
解决方案：
- 增加上下文数量
- 改进提示模板
- 使用更强的LLM

═══════════════════════════════════════════════════════════════════
```

---

## 第九步：扩展练习

### 9.1 练习1：添加网页爬取

```python
# 练习：实现网页爬取器
class WebScraper:
    def scrape_url(self, url: str) -> str:
        """
        爬取网页内容
        提示：使用requests和BeautifulSoup
        """
        pass
```

### 9.2 练习2：实现混合检索

```python
# 练习：结合关键词和语义搜索
class HybridRetriever:
    def search(self, query: str) -> List[Document]:
        """
        混合检索
        提示：结合BM25和向量搜索
        """
        pass
```

### 9.3 练习3：添加重排序

```python
# 练习：对检索结果重排序
class Reranker:
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        重排序
        提示：使用交叉编码器
        """
        pass
```

---

## 第十步：学习总结

```
学习收获
═══════════════════════════════════════════════════════════════════

通过完成本项目，你学会了：

✓ 理解RAG架构的核心组件
✓ 实现文档加载和分割
✓ 使用嵌入模型生成向量
✓ 构建向量存储和检索系统
✓ 集成LLM进行增强生成
✓ 实现多轮对话支持

关键概念：
- Document：文档数据结构
- TextSplitter：文本分割器
- Embedding：嵌入向量
- VectorStore：向量存储
- RAG：检索增强生成

核心洞察：
RAG = 检索 + 增强 + 生成
- 检索：找到相关文档
- 增强：注入上下文
- 生成：基于上下文回答

下一步：
1. 尝试加载自己的文档
2. 调整参数优化效果
3. 继续实践项目4：多代理协作

═══════════════════════════════════════════════════════════════════
```

---

## 参考资源

```
参考资源
═══════════════════════════════════════════════════════════════════

文档：
- ChromaDB文档：https://docs.trychroma.com/
- LangChain RAG：https://python.langchain.com/docs/tutorials/rag/
- OpenAI嵌入：https://platform.openai.com/docs/guides/embeddings

论文：
- RAG: Lewis et al., "Retrieval-Augmented Generation" (2020)
- DPR: Karpukhin et al., "Dense Passage Retrieval" (2020)

相关项目：
- 实践项目1：简单ReAct代理
- 实践项目2：Reflexion代理
- 实践项目4：多代理协作

═══════════════════════════════════════════════════════════════════
```

---

*实践项目3完成于 2026-07-04*
*预计学习时间：3-4小时*
*难度：⭐⭐ 初级-中级*
