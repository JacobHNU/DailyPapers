# 实践项目1：构建简单的ReAct代理

> 新手友好教程 · 从零开始实现 · 完整代码示例
> 预计时间：2-3小时
> 难度：⭐ 入门

---

## 项目概述

```
项目目标
═══════════════════════════════════════════════════════════════════

构建一个简单的ReAct代理，能够：
1. 接收用户的自然语言问题
2. 进行推理（Thought）
3. 调用工具获取信息（Action）
4. 观察工具返回结果（Observation）
5. 循环直到得出最终答案

示例场景：
用户："北京今天天气怎么样？"
代理思考：我需要查询北京的天气信息
代理行动：调用天气API
观察结果：北京今天晴，25°C
代理思考：我已经获得天气信息，可以回答了
最终答案：北京今天天气晴朗，温度25°C

═══════════════════════════════════════════════════════════════════
```

---

## 第一步：环境准备

### 1.1 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv react_agent_env
source react_agent_env/bin/activate  # Linux/Mac
# 或
react_agent_env\Scripts\activate  # Windows

# 安装必要的包
pip install openai langchain langchain-openai python-dotenv requests
```

### 1.2 配置API密钥

```python
# 创建 .env 文件
# 文件内容：
OPENAI_API_KEY=your_api_key_here

# 或者直接在代码中设置（不推荐用于生产）
import os
os.environ["OPENAI_API_KEY"] = "your_api_key_here"
```

---

## 第二步：理解ReAct框架

### 2.1 ReAct核心概念

```
ReAct循环
═══════════════════════════════════════════════════════════════════

Thought（思考）：推理当前状态，决定下一步
    ↓
Action（行动）：调用工具获取信息
    ↓
Observation（观察）：获取工具返回结果
    ↓
重复循环直到得出最终答案

关键点：
- Thought是内部推理，帮助模型规划
- Action是实际的工具调用
- Observation是环境反馈
- 循环在模型决定"完成"时终止

═══════════════════════════════════════════════════════════════════
```

### 2.2 简化的ReAct提示模板

```python
REACT_PROMPT = """你是一个有用的助手。你可以使用以下工具来回答问题：

{tools}

请使用以下格式回答问题：

Question: 你需要回答的问题
Thought: 你应该思考该怎么做
Action: 你要使用的工具名称
Action Input: 工具的输入参数
Observation: 工具返回的结果
... (这个Thought/Action/Action Input/Observation可以重复多次)
Thought: 我现在知道最终答案了
Final Answer: 问题的最终答案

开始！

Question: {question}
Thought:"""
```

---

## 第三步：实现工具函数

### 3.1 定义工具

```python
# tools.py
import requests
from datetime import datetime

def search_wikipedia(query: str) -> str:
    """
    搜索维基百科获取信息
    
    参数:
        query: 搜索关键词
        
    返回:
        维基百科摘要文本
    """
    try:
        # 使用维基百科API
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("extract", "未找到相关信息")
        else:
            return f"搜索失败：HTTP {response.status_code}"
    except Exception as e:
        return f"搜索出错：{str(e)}"


def get_current_time() -> str:
    """
    获取当前时间
    
    返回:
        当前时间字符串
    """
    now = datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M:%S")


def calculator(expression: str) -> str:
    """
    计算数学表达式
    
    参数:
        expression: 数学表达式，如 "2 + 3 * 4"
        
    返回:
        计算结果
    """
    try:
        # 安全地计算表达式
        # 注意：生产环境应使用更安全的计算库
        result = eval(expression)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算错误：{str(e)}"


def search_weather(city: str) -> str:
    """
    查询城市天气（模拟）
    
    参数:
        city: 城市名称
        
    返回:
        天气信息
    """
    # 这里使用模拟数据，实际应用可接入天气API
    weather_data = {
        "北京": "北京今天晴朗，温度25°C，湿度40%",
        "上海": "上海今天多云，温度28°C，湿度65%",
        "广州": "广州今天小雨，温度30°C，湿度80%",
        "深圳": "深圳今天阴天，温度29°C，湿度70%",
    }
    
    return weather_data.get(city, f"未找到{city}的天气信息")


# 工具注册表
TOOLS = {
    "search_wikipedia": {
        "func": search_wikipedia,
        "description": "搜索维基百科获取信息。输入应该是搜索关键词。"
    },
    "get_current_time": {
        "func": get_current_time,
        "description": "获取当前时间。不需要输入参数。"
    },
    "calculator": {
        "func": calculator,
        "description": "计算数学表达式。输入应该是数学表达式，如'2 + 3 * 4'。"
    },
    "search_weather": {
        "func": search_weather,
        "description": "查询城市天气。输入应该是城市名称。"
    }
}
```

### 3.2 格式化工具描述

```python
# utils.py

def format_tools_description(tools: dict) -> str:
    """
    将工具字典格式化为提示文本
    
    参数:
        tools: 工具字典
        
    返回:
        格式化的工具描述
    """
    descriptions = []
    for name, tool_info in tools.items():
        desc = f"- {name}: {tool_info['description']}"
        descriptions.append(desc)
    
    return "\n".join(descriptions)


def parse_action(text: str) -> tuple:
    """
    从模型输出中解析Action和Action Input
    
    参数:
        text: 模型输出文本
        
    返回:
        (action, action_input) 元组
    """
    lines = text.strip().split("\n")
    
    action = None
    action_input = None
    
    for line in lines:
        if line.startswith("Action:"):
            action = line.replace("Action:", "").strip()
        elif line.startswith("Action Input:"):
            action_input = line.replace("Action Input:", "").strip()
    
    return action, action_input
```

---

## 第四步：实现ReAct代理核心

### 4.1 简单版本（使用OpenAI API）

```python
# react_agent_simple.py
import os
from openai import OpenAI
from dotenv import load_dotenv

from tools import TOOLS
from utils import format_tools_description, parse_action

# 加载环境变量
load_dotenv()

class SimpleReActAgent:
    """
    简单的ReAct代理实现
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        初始化代理
        
        参数:
            model: 使用的模型名称
        """
        self.client = OpenAI()
        self.model = model
        self.tools = TOOLS
        self.max_iterations = 5  # 最大循环次数
        
    def run(self, question: str) -> str:
        """
        运行代理回答问题
        
        参数:
            question: 用户问题
            
        返回:
            最终答案
        """
        # 构建工具描述
        tools_desc = format_tools_description(self.tools)
        
        # 构建初始提示
        prompt = f"""你是一个有用的助手。你可以使用以下工具来回答问题：

{tools_desc}

请使用以下格式回答问题：

Question: 你需要回答的问题
Thought: 你应该思考该怎么做
Action: 你要使用的工具名称
Action Input: 工具的输入参数
Observation: 工具返回的结果
... (这个Thought/Action/Action Input/Observation可以重复多次)
Thought: 我现在知道最终答案了
Final Answer: 问题的最终答案

开始！

Question: {question}
Thought:"""
        
        # 开始ReAct循环
        for iteration in range(self.max_iterations):
            print(f"\n--- 迭代 {iteration + 1} ---")
            
            # 调用LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=500
            )
            
            # 获取模型输出
            model_output = response.choices[0].message.content
            print(f"模型输出:\n{model_output}")
            
            # 检查是否包含Final Answer
            if "Final Answer:" in model_output:
                # 提取最终答案
                final_answer = model_output.split("Final Answer:")[-1].strip()
                return final_answer
            
            # 解析Action和Action Input
            action, action_input = parse_action(model_output)
            
            if action is None:
                # 如果没有Action，可能是模型输出格式错误
                return f"错误：无法解析模型输出。原始输出：{model_output}"
            
            # 执行工具
            if action in self.tools:
                tool_func = self.tools[action]["func"]
                if action_input:
                    observation = tool_func(action_input)
                else:
                    observation = tool_func()
                
                print(f"执行工具: {action}")
                print(f"工具输入: {action_input}")
                print(f"工具输出: {observation}")
            else:
                observation = f"错误：未找到工具 '{action}'"
                print(f"工具错误: {observation}")
            
            # 将Observation添加到提示中
            prompt += f" {model_output}\nObservation: {observation}\nThought:"
        
        return "错误：达到最大迭代次数，未能得出最终答案"
```

### 4.2 测试简单代理

```python
# test_simple_agent.py
from react_agent_simple import SimpleReActAgent

def main():
    # 创建代理
    agent = SimpleReActAgent()
    
    # 测试问题
    questions = [
        "现在几点了？",
        "请计算 15 * 23 + 47",
        "北京今天天气怎么样？",
        "Python是什么？请用维基百科查询。",
    ]
    
    # 运行测试
    for question in questions:
        print(f"\n{'='*60}")
        print(f"问题: {question}")
        print(f"{'='*60}")
        
        answer = agent.run(question)
        
        print(f"\n最终答案: {answer}")

if __name__ == "__main__":
    main()
```

---

## 第五步：使用LangChain实现更完整的版本

### 5.1 LangChain版本

```python
# react_agent_langchain.py
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from tools import search_wikipedia, get_current_time, calculator, search_weather

def create_react_agent_langchain():
    """
    使用LangChain创建ReAct代理
    """
    
    # 1. 创建LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )
    
    # 2. 定义工具
    tools = [
        Tool(
            name="SearchWikipedia",
            func=search_wikipedia,
            description="搜索维基百科获取信息。输入应该是搜索关键词。"
        ),
        Tool(
            name="GetCurrentTime",
            func=lambda x: get_current_time(),
            description="获取当前时间。不需要输入参数，输入可以是空字符串。"
        ),
        Tool(
            name="Calculator",
            func=calculator,
            description="计算数学表达式。输入应该是数学表达式，如'2 + 3 * 4'。"
        ),
        Tool(
            name="SearchWeather",
            func=search_weather,
            description="查询城市天气。输入应该是城市名称。"
        ),
    ]
    
    # 3. 创建提示模板
    # LangChain的ReAct提示需要特定格式
    prompt = PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}""")
    
    # 4. 创建代理
    agent = create_react_agent(llm, tools, prompt)
    
    # 5. 创建代理执行器
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True
    )
    
    return agent_executor


def main():
    # 创建代理
    agent = create_react_agent_langchain()
    
    # 测试问题
    questions = [
        "现在几点了？",
        "请计算 15 * 23 + 47",
        "北京今天天气怎么样？",
        "Python是什么？请用维基百科查询。",
    ]
    
    # 运行测试
    for question in questions:
        print(f"\n{'='*60}")
        print(f"问题: {question}")
        print(f"{'='*60}")
        
        try:
            result = agent.invoke({"input": question})
            print(f"\n最终答案: {result['output']}")
        except Exception as e:
            print(f"\n错误: {str(e)}")


if __name__ == "__main__":
    main()
```

---

## 第六步：添加对话记忆

### 6.1 带记忆的ReAct代理

```python
# react_agent_with_memory.py
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from tools import search_wikipedia, get_current_time, calculator, search_weather

def create_react_agent_with_memory():
    """
    创建带对话记忆的ReAct代理
    """
    
    # 1. 创建LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )
    
    # 2. 定义工具
    tools = [
        Tool(
            name="SearchWikipedia",
            func=search_wikipedia,
            description="搜索维基百科获取信息。输入应该是搜索关键词。"
        ),
        Tool(
            name="GetCurrentTime",
            func=lambda x: get_current_time(),
            description="获取当前时间。不需要输入参数，输入可以是空字符串。"
        ),
        Tool(
            name="Calculator",
            func=calculator,
            description="计算数学表达式。输入应该是数学表达式，如'2 + 3 * 4'。"
        ),
        Tool(
            name="SearchWeather",
            func=search_weather,
            description="查询城市天气。输入应该是城市名称。"
        ),
    ]
    
    # 3. 创建提示模板（包含对话历史）
    prompt = PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Previous conversation history:
{chat_history}

Begin!

Question: {input}
Thought:{agent_scratchpad}""")
    
    # 4. 创建记忆
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # 5. 创建代理
    agent = create_react_agent(llm, tools, prompt)
    
    # 6. 创建代理执行器（带记忆）
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True
    )
    
    return agent_executor


def main():
    # 创建带记忆的代理
    agent = create_react_agent_with_memory()
    
    # 模拟多轮对话
    conversations = [
        "北京今天天气怎么样？",
        "上海呢？",
        "这两个城市哪个更热？",
        "现在几点了？",
        "我们之前讨论了哪些城市？",
    ]
    
    print("开始多轮对话测试...")
    
    for i, question in enumerate(conversations, 1):
        print(f"\n{'='*60}")
        print(f"轮次 {i}: {question}")
        print(f"{'='*60}")
        
        try:
            result = agent.invoke({"input": question})
            print(f"\n回答: {result['output']}")
        except Exception as e:
            print(f"\n错误: {str(e)}")
        
        print(f"\n{'─'*40}")
    
    # 查看记忆
    print("\n\n对话历史:")
    print(agent.memory.buffer)


if __name__ == "__main__":
    main()
```

---

## 第七步：添加错误处理和日志

### 7.1 生产级ReAct代理

```python
# react_agent_production.py
import logging
from typing import Optional, Dict, Any
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from tools import search_wikipedia, get_current_time, calculator, search_weather

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionReActAgent:
    """
    生产级ReAct代理
    """
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0,
        max_iterations: int = 5,
        verbose: bool = False
    ):
        """
        初始化代理
        
        参数:
            model: 使用的模型
            temperature: 温度参数
            max_iterations: 最大迭代次数
            verbose: 是否显示详细日志
        """
        self.model = model
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # 创建组件
        self.llm = self._create_llm()
        self.tools = self._create_tools()
        self.prompt = self._create_prompt()
        self.memory = self._create_memory()
        
        # 创建代理
        self.agent = self._create_agent()
        
        logger.info(f"ReAct代理初始化完成，使用模型: {model}")
    
    def _create_llm(self) -> ChatOpenAI:
        """创建LLM"""
        return ChatOpenAI(
            model=self.model,
            temperature=self.temperature
        )
    
    def _create_tools(self) -> list:
        """创建工具列表"""
        return [
            Tool(
                name="SearchWikipedia",
                func=search_wikipedia,
                description="搜索维基百科获取信息。输入应该是搜索关键词。"
            ),
            Tool(
                name="GetCurrentTime",
                func=lambda x: get_current_time(),
                description="获取当前时间。不需要输入参数，输入可以是空字符串。"
            ),
            Tool(
                name="Calculator",
                func=calculator,
                description="计算数学表达式。输入应该是数学表达式，如'2 + 3 * 4'。"
            ),
            Tool(
                name="SearchWeather",
                func=search_weather,
                description="查询城市天气。输入应该是城市名称。"
            ),
        ]
    
    def _create_prompt(self) -> PromptTemplate:
        """创建提示模板"""
        return PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Previous conversation history:
{chat_history}

Begin!

Question: {input}
Thought:{agent_scratchpad}""")
    
    def _create_memory(self) -> ConversationBufferMemory:
        """创建对话记忆"""
        return ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    def _create_agent(self) -> AgentExecutor:
        """创建代理执行器"""
        agent = create_react_agent(self.llm, self.tools, self.prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=self.verbose,
            max_iterations=self.max_iterations,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
    
    def run(self, question: str) -> Dict[str, Any]:
        """
        运行代理
        
        参数:
            question: 用户问题
            
        返回:
            包含答案和中间步骤的字典
        """
        logger.info(f"收到问题: {question}")
        
        try:
            # 运行代理
            result = self.agent.invoke({"input": question})
            
            # 提取中间步骤
            intermediate_steps = []
            if "intermediate_steps" in result:
                for step in result["intermediate_steps"]:
                    action = step[0]
                    observation = step[1]
                    intermediate_steps.append({
                        "tool": action.tool,
                        "input": action.tool_input,
                        "output": observation
                    })
            
            logger.info(f"问题回答成功: {result['output'][:50]}...")
            
            return {
                "success": True,
                "answer": result["output"],
                "intermediate_steps": intermediate_steps
            }
            
        except Exception as e:
            logger.error(f"问题回答失败: {str(e)}")
            
            return {
                "success": False,
                "answer": f"抱歉，处理您的问题时出现错误: {str(e)}",
                "intermediate_steps": [],
                "error": str(e)
            }
    
    def clear_memory(self):
        """清除对话记忆"""
        self.memory.clear()
        logger.info("对话记忆已清除")
    
    def get_memory(self) -> list:
        """获取对话历史"""
        return self.memory.buffer


def main():
    # 创建生产级代理
    agent = ProductionReActAgent(
        model="gpt-3.5-turbo",
        verbose=True
    )
    
    # 测试问题
    test_questions = [
        "现在几点了？",
        "请计算 15 * 23 + 47",
        "北京今天天气怎么样？",
        "Python是什么？请用维基百科查询。",
    ]
    
    print("开始测试生产级ReAct代理...")
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"问题: {question}")
        print(f"{'='*60}")
        
        result = agent.run(question)
        
        if result["success"]:
            print(f"\n✓ 回答成功")
            print(f"答案: {result['answer']}")
            
            if result["intermediate_steps"]:
                print(f"\n中间步骤:")
                for i, step in enumerate(result["intermediate_steps"], 1):
                    print(f"  {i}. 工具: {step['tool']}")
                    print(f"     输入: {step['input']}")
                    print(f"     输出: {step['output'][:100]}...")
        else:
            print(f"\n✗ 回答失败")
            print(f"错误: {result.get('error', '未知错误')}")
    
    # 清除记忆
    agent.clear_memory()


if __name__ == "__main__":
    main()
```

---

## 第八步：完整项目结构

### 8.1 推荐的项目结构

```
react_agent_project/
├── .env                    # 环境变量（API密钥）
├── requirements.txt        # 依赖列表
├── README.md              # 项目说明
│
├── tools/                 # 工具模块
│   ├── __init__.py
│   ├── search.py          # 搜索工具
│   ├── calculator.py      # 计算工具
│   ├── weather.py         # 天气工具
│   └── time_tool.py       # 时间工具
│
├── agents/                # 代理模块
│   ├── __init__.py
│   ├── react_agent.py     # ReAct代理实现
│   └── memory.py          # 记忆管理
│
├── utils/                 # 工具函数
│   ├── __init__.py
│   ├── prompt.py          # 提示模板
│   └── parser.py          # 输出解析
│
├── tests/                 # 测试文件
│   ├── test_tools.py
│   └── test_agent.py
│
├── examples/              # 示例脚本
│   ├── simple_example.py
│   └── advanced_example.py
│
└── main.py               # 主入口
```

### 8.2 requirements.txt

```
openai>=1.0.0
langchain>=0.1.0
langchain-openai>=0.0.5
python-dotenv>=1.0.0
requests>=2.31.0
```

---

## 第九步：运行和测试

### 9.1 运行简单示例

```bash
# 进入项目目录
cd react_agent_project

# 运行简单示例
python examples/simple_example.py
```

### 9.2 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_agent.py -v
```

### 9.3 交互式运行

```python
# interactive.py
from agents.react_agent import ProductionReActAgent

def main():
    agent = ProductionReActAgent(verbose=False)
    
    print("ReAct代理已启动！输入 'quit' 退出。")
    print("="*50)
    
    while True:
        question = input("\n请输入您的问题: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("再见！")
            break
        
        if not question:
            continue
        
        result = agent.run(question)
        
        if result["success"]:
            print(f"\n回答: {result['answer']}")
        else:
            print(f"\n错误: {result['answer']}")

if __name__ == "__main__":
    main()
```

---

## 第十步：扩展和改进

### 10.1 添加更多工具

```python
# 添加网络搜索工具
def search_web(query: str) -> str:
    """
    搜索网络获取信息
    """
    # 可以使用SerpAPI、Bing API等
    # 这里使用示例实现
    pass

# 添加文件读取工具
def read_file(file_path: str) -> str:
    """
    读取文件内容
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"读取文件失败: {str(e)}"
```

### 10.2 添加流式输出

```python
def run_streaming(self, question: str):
    """
    流式运行代理
    """
    # 使用LangChain的流式功能
    for chunk in self.agent.stream({"input": question}):
        yield chunk
```

### 10.3 添加并发支持

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncReActAgent:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=5)
    
    async def arun(self, question: str) -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.run,
            question
        )
```

---

## 常见问题和解决方案

```
常见问题
═══════════════════════════════════════════════════════════════════

问题1：模型输出格式不正确
解决方案：
- 更明确的提示
- 添加格式验证
- 使用更强大的模型（GPT-4）

问题2：工具调用失败
解决方案：
- 添加错误处理
- 重试机制
- 降级策略

问题3：无限循环
解决方案：
- 设置最大迭代次数
- 检测重复动作
- 超时机制

问题4：记忆溢出
解决方案：
- 限制记忆大小
- 摘要压缩
- 定期清理

问题5：API限流
解决方案：
- 添加延迟
- 使用退避策略
- 缓存结果

═══════════════════════════════════════════════════════════════════
```

---

## 下一步学习

```
学习路径
═══════════════════════════════════════════════════════════════════

完成本项目后，可以继续：

1. 实践项目2：实现Reflexion代理
   - 学习从失败中改进
   - 实现语言反思机制

2. 实践项目3：构建RAG记忆系统
   - 学习检索增强生成
   - 实现向量存储

3. 实践项目4：多代理协作
   - 学习代理间通信
   - 实现任务分解

4. 实践项目5：代理轨迹训练
   - 学习RL训练代理
   - 实现GRPO/DPO

═══════════════════════════════════════════════════════════════════
```

---

## 总结

```
项目收获
═══════════════════════════════════════════════════════════════════

通过完成本项目，你将学会：

✓ 理解ReAct框架的核心概念
✓ 实现基本的工具调用机制
✓ 构建简单的推理-行动循环
✓ 使用LangChain构建代理
✓ 添加对话记忆功能
✓ 实现错误处理和日志
✓ 组织生产级代码结构

关键概念：
- Thought：内部推理
- Action：工具调用
- Observation：环境反馈
- 循环：持续改进

下一步：
尝试添加更多工具，测试更复杂的问题，
然后继续下一个实践项目！

═══════════════════════════════════════════════════════════════════
```

---

*实践项目1完成于 2026-07-04*
*预计学习时间：2-3小时*
*难度：⭐ 入门*
