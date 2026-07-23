# 实践项目2：实现Reflexion代理

> 新手友好教程 · 学习从失败中改进 · 语言反思机制
> 预计时间：3-4小时
> 难度：⭐⭐ 初级-中级

---

## 项目概述

```
项目目标
═══════════════════════════════════════════════════════════════════

实现一个Reflexion代理，能够：
1. 尝试解决问题
2. 从失败中学习
3. 生成自然语言反思
4. 在下次尝试时应用反思
5. 逐步改进直到成功

核心思想：RL without weight updates
- 不更新模型权重
- 通过语言自我批评改进
- 反思存储在情景记忆中
- 下次尝试时注入提示

示例场景：
任务：编写一个函数计算斐波那契数列
尝试1：函数有bug，测试失败
反思：我忘记了处理n=0和n=1的边界情况
尝试2：添加边界情况处理，测试通过

═══════════════════════════════════════════════════════════════════
```

---

## 第一步：理解Reflexion架构

### 1.1 Reflexion四个组件

```
Reflexion架构
═══════════════════════════════════════════════════════════════════

1. Actor（演员）
   - LLM代理在环境中执行动作
   - 生成代码/动作

2. Evaluator（评估器）
   - 评估执行结果
   - 返回成功/失败信号
   - 可以是测试执行、代码检查等

3. Self-Reflection Generator（自我反思生成器）
   - 给定失败的尝试和错误信息
   - 生成自然语言反思
   - 分析失败原因

4. Episodic Memory（情景记忆）
   - 存储过去的反思
   - 下次尝试时注入提示
   - 通常限制为最近3-5个反思

═══════════════════════════════════════════════════════════════════
```

### 1.2 Reflexion流程

```
Reflexion流程
═══════════════════════════════════════════════════════════════════

循环开始：
1. Actor尝试解决任务
2. Evaluator评估结果
3. 如果成功 → 结束
4. 如果失败 → 生成反思
5. 存储反思到情景记忆
6. 重复循环（带反思上下文）

关键：每次尝试都会携带之前的反思
代理从自己的错误中学习

═══════════════════════════════════════════════════════════════════
```

---

## 第二步：环境准备

### 2.1 安装依赖

```bash
# 创建虚拟环境
python -m venv reflexion_env
source reflexion_env/bin/activate  # Linux/Mac
# 或
reflexion_env\Scripts\activate  # Windows

# 安装依赖
pip install openai langchain langchain-openai python-dotenv
```

### 2.2 项目结构

```
reflexion_project/
├── .env
├── requirements.txt
├── README.md
│
├── agents/
│   ├── __init__.py
│   ├── reflexion_agent.py    # Reflexion代理核心
│   └── memory.py             # 情景记忆
│
├── evaluators/
│   ├── __init__.py
│   ├── code_evaluator.py     # 代码评估器
│   └── task_evaluator.py     # 任务评估器
│
├── prompts/
│   ├── __init__.py
│   ├── actor_prompt.py       # Actor提示
│   └── reflection_prompt.py  # 反思提示
│
├── examples/
│   ├── coding_example.py     # 代码任务示例
│   └── reasoning_example.py  # 推理任务示例
│
└── main.py
```

---

## 第三步：实现核心组件

### 3.1 情景记忆

```python
# agents/memory.py
from typing import List, Optional
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Reflection:
    """反思数据类"""
    attempt_number: int
    task: str
    code: str
    error: str
    reflection: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_context_string(self) -> str:
        """转换为上下文字符串"""
        return f"""[尝试 {self.attempt_number}]
任务: {self.task}
代码: {self.code[:200]}...
错误: {self.error}
反思: {self.reflection}
"""


class EpisodicMemory:
    """
    情景记忆
    存储和管理反思历史
    """
    
    def __init__(self, max_reflections: int = 5):
        """
        初始化情景记忆
        
        参数:
            max_reflections: 最大存储反思数量
        """
        self.reflections: List[Reflection] = []
        self.max_reflections = max_reflections
    
    def add_reflection(self, reflection: Reflection):
        """
        添加反思
        
        参数:
            reflection: 反思对象
        """
        self.reflections.append(reflection)
        
        # 超过最大数量时，移除最旧的
        if len(self.reflections) > self.max_reflections:
            self.reflections = self.reflections[-self.max_reflections:]
    
    def get_context(self) -> str:
        """
        获取反思上下文
        用于注入到下次尝试的提示中
        
        返回:
            格式化的反思上下文
        """
        if not self.reflections:
            return "暂无先前尝试的反思。"
        
        context_parts = ["=== 来自先前尝试的反思 ===\n"]
        
        for reflection in self.reflections:
            context_parts.append(reflection.to_context_string())
        
        context_parts.append("=== 反思结束 ===\n")
        context_parts.append("请从这些反思中学习，避免重复相同的错误。\n")
        
        return "\n".join(context_parts)
    
    def clear(self):
        """清除所有反思"""
        self.reflections = []
    
    def get_reflection_count(self) -> int:
        """获取反思数量"""
        return len(self.reflections)
```

### 3.2 代码评估器

```python
# evaluators/code_evaluator.py
import subprocess
import tempfile
import os
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class EvalResult:
    """评估结果"""
    success: bool
    output: str
    error: Optional[str] = None
    test_results: Optional[dict] = None


class CodeEvaluator:
    """
    代码评估器
    执行代码并验证结果
    """
    
    def __init__(self, timeout: int = 10):
        """
        初始化评估器
        
        参数:
            timeout: 执行超时时间（秒）
        """
        self.timeout = timeout
    
    def evaluate_python_code(
        self, 
        code: str, 
        test_cases: list
    ) -> EvalResult:
        """
        评估Python代码
        
        参数:
            code: 要评估的代码
            test_cases: 测试用例列表，每个元素是 (input, expected_output)
            
        返回:
            评估结果
        """
        # 构建完整的测试代码
        test_code = self._build_test_code(code, test_cases)
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.py', 
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(test_code)
            temp_file = f.name
        
        try:
            # 执行代码
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                return EvalResult(
                    success=True,
                    output=result.stdout,
                    test_results={"passed": len(test_cases), "total": len(test_cases)}
                )
            else:
                return EvalResult(
                    success=False,
                    output=result.stdout,
                    error=result.stderr
                )
                
        except subprocess.TimeoutExpired:
            return EvalResult(
                success=False,
                output="",
                error=f"代码执行超时（超过{self.timeout}秒）"
            )
        except Exception as e:
            return EvalResult(
                success=False,
                output="",
                error=f"执行错误: {str(e)}"
            )
        finally:
            # 清理临时文件
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def _build_test_code(self, code: str, test_cases: list) -> str:
        """
        构建包含测试的完整代码
        
        参数:
            code: 用户代码
            test_cases: 测试用例
            
        返回:
            完整的测试代码
        """
        test_code = f"""{code}

# 测试代码
test_results = []
"""
        
        for i, (input_val, expected) in enumerate(test_cases, 1):
            test_code += f"""
try:
    result = {input_val}
    expected = {expected}
    if result == expected:
        test_results.append(("Test {i}", "PASS"))
    else:
        test_results.append(("Test {i}", f"FAIL: expected {{expected}}, got {{result}}"))
except Exception as e:
    test_results.append(("Test {i}", f"ERROR: {{str(e)}}"))
"""
        
        test_code += """
# 打印结果
for name, result in test_results:
    print(f"{name}: {result}")
"""
        
        return test_code
    
    def evaluate_function(
        self,
        function_code: str,
        function_name: str,
        test_cases: list
    ) -> EvalResult:
        """
        评估函数定义
        
        参数:
            function_code: 函数代码
            function_name: 函数名称
            test_cases: 测试用例列表，每个元素是 (args, expected)
            
        返回:
            评估结果
        """
        # 构建测试代码
        test_code = f"""{function_code}

# 测试函数
test_results = []
"""
        
        for i, (args, expected) in enumerate(test_cases, 1):
            if isinstance(args, tuple):
                args_str = ", ".join(repr(a) for a in args)
            else:
                args_str = repr(args)
            
            test_code += f"""
try:
    result = {function_name}({args_str})
    expected = {expected}
    if result == expected:
        test_results.append(("Test {i}", "PASS"))
    else:
        test_results.append(("Test {i}", f"FAIL: expected {{expected}}, got {{result}}"))
except Exception as e:
    test_results.append(("Test {i}", f"ERROR: {{str(e)}}"))
"""
        
        test_code += """
# 打印结果
for name, result in test_results:
    print(f"{name}: {result}")
"""
        
        return self.evaluate_python_code(test_code, [])
```

### 3.3 反思生成提示

```python
# prompts/reflection_prompt.py

REFLECTION_PROMPT = """你是一个帮助AI代理从失败中学习的助手。

代理尝试完成以下任务但失败了。

任务: {task}

代理生成的代码:
```python
{code}
```

执行错误:
{error}

请分析失败的原因，并生成一个简洁的反思，帮助代理在下次尝试时避免相同的错误。

反思应该：
1. 明确指出错误的根本原因
2. 提供具体的改进建议
3. 简洁明了（不超过3-4句话）

反思:"""

ACTOR_PROMPT_WITH_REFLECTIONS = """你是一个编程助手。请完成以下任务。

任务: {task}

{reflections_context}

要求：
1. 仔细阅读任务要求
2. 从先前的反思中学习，避免重复错误
3. 编写正确、完整的代码
4. 处理边界情况

请直接输出代码，不要包含其他解释。

```python
"""

ACTOR_PROMPT_WITHOUT_REFLECTIONS = """你是一个编程助手。请完成以下任务。

任务: {task}

要求：
1. 仔细阅读任务要求
2. 编写正确、完整的代码
3. 处理边界情况

请直接输出代码，不要包含其他解释。

```python
"""
```

---

## 第四步：实现Reflexion代理

### 4.1 Reflexion代理核心

```python
# agents/reflexion_agent.py
import os
from typing import Optional, Tuple
from openai import OpenAI
from dotenv import load_dotenv

from agents.memory import EpisodicMemory, Reflection
from evaluators.code_evaluator import CodeEvaluator, EvalResult
from prompts.reflection_prompt import (
    REFLECTION_PROMPT,
    ACTOR_PROMPT_WITH_REFLECTIONS,
    ACTOR_PROMPT_WITHOUT_REFLECTIONS
)

load_dotenv()


class ReflexionAgent:
    """
    Reflexion代理
    通过语言反思从失败中学习
    """
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        max_attempts: int = 5,
        max_reflections: int = 3,
        verbose: bool = False
    ):
        """
        初始化Reflexion代理
        
        参数:
            model: 使用的模型
            max_attempts: 最大尝试次数
            max_reflections: 最大反思数量
            verbose: 是否显示详细信息
        """
        self.client = OpenAI()
        self.model = model
        self.max_attempts = max_attempts
        self.verbose = verbose
        
        # 初始化组件
        self.memory = EpisodicMemory(max_reflections=max_reflections)
        self.evaluator = CodeEvaluator()
    
    def solve(self, task: str, test_cases: list) -> dict:
        """
        解决编程任务
        
        参数:
            task: 任务描述
            test_cases: 测试用例列表
            
        返回:
            包含结果的字典
        """
        print(f"\n{'='*60}")
        print(f"开始解决任务: {task}")
        print(f"{'='*60}")
        
        # 清除之前的反思
        self.memory.clear()
        
        for attempt in range(1, self.max_attempts + 1):
            print(f"\n--- 尝试 {attempt} ---")
            
            # 1. Actor生成代码
            code = self._generate_code(task)
            print(f"\n生成的代码:\n{code}")
            
            # 2. 评估代码
            eval_result = self.evaluator.evaluate_function(
                function_code=code,
                function_name=self._extract_function_name(code),
                test_cases=test_cases
            )
            
            print(f"\n评估结果:")
            print(f"  成功: {eval_result.success}")
            if eval_result.error:
                print(f"  错误: {eval_result.error}")
            
            # 3. 检查是否成功
            if eval_result.success:
                print(f"\n✓ 任务成功完成！（尝试 {attempt}）")
                return {
                    "success": True,
                    "code": code,
                    "attempts": attempt,
                    "reflections": self.memory.get_reflection_count()
                }
            
            # 4. 生成反思
            print(f"\n任务失败，生成反思...")
            reflection_text = self._generate_reflection(
                task=task,
                code=code,
                error=eval_result.error or "测试未通过"
            )
            
            print(f"反思: {reflection_text}")
            
            # 5. 存储反思
            reflection = Reflection(
                attempt_number=attempt,
                task=task,
                code=code,
                error=eval_result.error or "测试未通过",
                reflection=reflection_text
            )
            self.memory.add_reflection(reflection)
        
        # 达到最大尝试次数
        print(f"\n✗ 达到最大尝试次数 ({self.max_attempts})")
        return {
            "success": False,
            "code": code,
            "attempts": self.max_attempts,
            "reflections": self.memory.get_reflection_count()
        }
    
    def _generate_code(self, task: str) -> str:
        """
        生成代码
        
        参数:
            task: 任务描述
            
        返回:
            生成的代码
        """
        # 获取反思上下文
        reflections_context = self.memory.get_context()
        
        # 构建提示
        if self.memory.get_reflection_count() > 0:
            prompt = ACTOR_PROMPT_WITH_REFLECTIONS.format(
                task=task,
                reflections_context=reflections_context
            )
        else:
            prompt = ACTOR_PROMPT_WITHOUT_REFLECTIONS.format(
                task=task
            )
        
        # 调用LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        
        # 提取代码
        code = response.choices[0].message.content
        
        # 清理代码（移除可能的markdown标记）
        code = self._clean_code(code)
        
        return code
    
    def _generate_reflection(
        self, 
        task: str, 
        code: str, 
        error: str
    ) -> str:
        """
        生成反思
        
        参数:
            task: 任务描述
            code: 生成的代码
            error: 错误信息
            
        返回:
            反思文本
        """
        prompt = REFLECTION_PROMPT.format(
            task=task,
            code=code,
            error=error
        )
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        return response.choices[0].message.content.strip()
    
    def _extract_function_name(self, code: str) -> str:
        """
        从代码中提取函数名
        
        参数:
            code: 代码文本
            
        返回:
            函数名
        """
        for line in code.split('\n'):
            if line.strip().startswith('def '):
                # 提取函数名
                func_name = line.split('def ')[1].split('(')[0].strip()
                return func_name
        
        # 默认函数名
        return "solution"
    
    def _clean_code(self, code: str) -> str:
        """
        清理代码
        
        参数:
            code: 原始代码
            
        返回:
            清理后的代码
        """
        # 移除可能的markdown代码块标记
        if code.startswith("```python"):
            code = code[9:]
        if code.startswith("```"):
            code = code[3:]
        if code.endswith("```"):
            code = code[:-3]
        
        return code.strip()
```

---

## 第五步：测试Reflexion代理

### 5.1 编程任务测试

```python
# examples/coding_example.py
from agents.reflexion_agent import ReflexionAgent

def test_fibonacci():
    """
    测试斐波那契数列任务
    """
    agent = ReflexionAgent(
        model="gpt-3.5-turbo",
        max_attempts=5,
        verbose=True
    )
    
    task = "编写一个Python函数fibonacci(n)，返回斐波那契数列的第n项。fibonacci(0)=0, fibonacci(1)=1"
    
    test_cases = [
        ((0,), 0),
        ((1,), 1),
        ((2,), 1),
        ((5,), 5),
        ((10,), 55),
        ((20,), 6765),
    ]
    
    result = agent.solve(task, test_cases)
    
    print(f"\n{'='*60}")
    print(f"最终结果:")
    print(f"  成功: {result['success']}")
    print(f"  尝试次数: {result['attempts']}")
    print(f"  反思次数: {result['reflections']}")
    if result['success']:
        print(f"  最终代码:\n{result['code']}")


def test_reverse_string():
    """
    测试字符串反转任务
    """
    agent = ReflexionAgent(
        model="gpt-3.5-turbo",
        max_attempts=5,
        verbose=True
    )
    
    task = "编写一个Python函数reverse_string(s)，返回字符串s的反转。处理空字符串和None的情况。"
    
    test_cases = [
        (("hello",), "olleh"),
        (("",), ""),
        (("a",), "a"),
        (("racecar",), "racecar"),
        (("Python",), "nohtyP"),
    ]
    
    result = agent.solve(task, test_cases)
    
    print(f"\n{'='*60}")
    print(f"最终结果:")
    print(f"  成功: {result['success']}")
    print(f"  尝试次数: {result['attempts']}")


def test_binary_search():
    """
    测试二分搜索任务（更复杂）
    """
    agent = ReflexionAgent(
        model="gpt-3.5-turbo",
        max_attempts=5,
        verbose=True
    )
    
    task = """编写一个Python函数binary_search(arr, target)，实现二分搜索。
- 如果找到target，返回其索引
- 如果未找到，返回-1
- 处理空数组的情况
- 数组已排序"""
    
    test_cases = [
        (([1, 2, 3, 4, 5], 3), 2),
        (([1, 2, 3, 4, 5], 1), 0),
        (([1, 2, 3, 4, 5], 5), 4),
        (([1, 2, 3, 4, 5], 6), -1),
        (([], 1), -1),
        (([1], 1), 0),
        (([1], 2), -1),
    ]
    
    result = agent.solve(task, test_cases)
    
    print(f"\n{'='*60}")
    print(f"最终结果:")
    print(f"  成功: {result['success']}")
    print(f"  尝试次数: {result['attempts']}")


if __name__ == "__main__":
    print("="*60)
    print("测试1: 斐波那契数列")
    print("="*60)
    test_fibonacci()
    
    print("\n\n")
    print("="*60)
    print("测试2: 字符串反转")
    print("="*60)
    test_reverse_string()
    
    print("\n\n")
    print("="*60)
    print("测试3: 二分搜索")
    print("="*60)
    test_binary_search()
```

### 5.2 运行测试

```bash
# 运行所有示例
python examples/coding_example.py

# 或运行特定示例
python -c "from examples.coding_example import test_fibonacci; test_fibonacci()"
```

---

## 第六步：增强功能

### 6.1 添加重试和退避机制

```python
# agents/reflexion_agent_enhanced.py
import time
from typing import Optional
from openai import OpenAI, APIError, RateLimitError

class EnhancedReflexionAgent(ReflexionAgent):
    """
    增强版Reflexion代理
    添加重试和退避机制
    """
    
    def __init__(self, *args, max_retries: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_retries = max_retries
    
    def _call_llm_with_retry(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        带重试的LLM调用
        
        参数:
            prompt: 提示文本
            max_tokens: 最大令牌数
            
        返回:
            模型输出
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
                
            except RateLimitError as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # 指数退避
                    print(f"API限流，等待{wait_time}秒后重试...")
                    time.sleep(wait_time)
                else:
                    raise e
                    
            except APIError as e:
                if attempt < self.max_retries - 1:
                    print(f"API错误，重试中...")
                    time.sleep(1)
                else:
                    raise e
        
        raise Exception("达到最大重试次数")
```

### 6.2 添加详细日志

```python
# agents/reflexion_agent_with_logging.py
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReflexionAgentWithLogging(ReflexionAgent):
    """
    带详细日志的Reflexion代理
    """
    
    def solve(self, task: str, test_cases: list) -> dict:
        logger.info(f"开始解决任务: {task}")
        logger.info(f"测试用例数量: {len(test_cases)}")
        
        # 记录开始时间
        start_time = datetime.now()
        
        # 调用父类方法
        result = super().solve(task, test_cases)
        
        # 记录结束时间
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"任务完成: 成功={result['success']}, "
                    f"尝试次数={result['attempts']}, "
                    f"耗时={duration:.2f}秒")
        
        return result
```

### 6.3 添加可视化

```python
# utils/visualization.py
from typing import List
from agents.memory import Reflection

def visualize_reflection_journey(reflections: List[Reflection]):
    """
    可视化反思历程
    
    参数:
        reflections: 反思列表
    """
    print("\n" + "="*60)
    print("反思历程可视化")
    print("="*60)
    
    for i, reflection in enumerate(reflections, 1):
        print(f"\n尝试 {i}:")
        print(f"  错误: {reflection.error[:50]}...")
        print(f"  反思: {reflection.reflection[:100]}...")
        
        if i < len(reflections):
            print("  ↓")
    
    print(f"\n总共 {len(reflections)} 次反思")

def generate_reflection_summary(reflections: List[Reflection]) -> str:
    """
    生成反思摘要
    
    参数:
        reflections: 反思列表
        
    返回:
        摘要文本
    """
    if not reflections:
        return "无反思记录"
    
    summary_parts = ["反思摘要:\n"]
    
    for i, reflection in enumerate(reflections, 1):
        summary_parts.append(f"{i}. {reflection.reflection[:100]}...")
    
    return "\n".join(summary_parts)
```

---

## 第七步：完整示例

### 7.1 主入口

```python
# main.py
from agents.reflexion_agent import ReflexionAgent
from examples.coding_example import (
    test_fibonacci,
    test_reverse_string,
    test_binary_search
)

def main():
    print("Reflexion代理实践项目")
    print("="*60)
    
    # 选择测试
    tests = {
        "1": ("斐波那契数列", test_fibonacci),
        "2": ("字符串反转", test_reverse_string),
        "3": ("二分搜索", test_binary_search),
    }
    
    print("\n可用测试:")
    for key, (name, _) in tests.items():
        print(f"  {key}. {name}")
    
    choice = input("\n选择测试 (1-3, 或 'all' 运行所有): ").strip()
    
    if choice == 'all':
        for name, test_func in tests.values():
            print(f"\n\n{'='*60}")
            print(f"测试: {name}")
            print('='*60)
            test_func()
    elif choice in tests:
        name, test_func = tests[choice]
        print(f"\n运行测试: {name}")
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

问题1：反思质量不高
症状：代理重复相同的错误
解决方案：
- 使用更强的模型（GPT-4）
- 改进反思提示
- 添加更多上下文

问题2：达到最大尝试次数
症状：代理无法在限制内解决
解决方案：
- 增加最大尝试次数
- 简化任务
- 提供更多示例

问题3：代码执行超时
症状：评估器返回超时错误
解决方案：
- 增加超时时间
- 优化代码生成
- 检查无限循环

问题4：API限流
症状：频繁的RateLimitError
解决方案：
- 添加重试机制
- 使用指数退避
- 考虑使用API池

问题5：反思不具体
症状：反思太笼统，无法指导改进
解决方案：
- 要求更具体的反思
- 包含代码行号
- 提供修复建议

═══════════════════════════════════════════════════════════════════
```

---

## 第九步：扩展练习

### 9.1 练习1：添加更多评估器

```python
# 练习：实现一个推理任务评估器
class ReasoningEvaluator:
    def evaluate(self, answer: str, expected: str) -> EvalResult:
        """
        评估推理答案
        提示：考虑语义相似度，而非精确匹配
        """
        pass
```

### 9.2 练习2：实现多轮对话

```python
# 练习：让代理能够处理澄清问题
class ConversationalReflexionAgent:
    def solve_with_clarification(self, task: str) -> dict:
        """
        如果任务不清晰，代理应该能够提问澄清
        提示：添加一个判断任务是否清晰的步骤
        """
        pass
```

### 9.3 练习3：添加缓存

```python
# 练习：缓存成功的解决方案
class CachingReflexionAgent:
    def __init__(self):
        self.cache = {}  # 任务 -> 解决方案
    
    def solve(self, task: str, test_cases: list) -> dict:
        """
        先检查缓存，如果找到相似任务，直接使用缓存的解决方案
        提示：使用嵌入相似度比较任务
        """
        pass
```

---

## 第十步：学习总结

```
学习收获
═══════════════════════════════════════════════════════════════════

通过完成本项目，你学会了：

✓ 理解Reflexion框架的核心思想
✓ 实现情景记忆系统
✓ 构建代码评估器
✓ 生成自然语言反思
✓ 实现迭代改进循环

关键概念：
- Actor：执行动作的代理
- Evaluator：评估结果的组件
- Reflection：从失败中学习的机制
- Episodic Memory：存储反思的内存

核心洞察：
RL without weight updates
- 不需要梯度下降
- 通过语言反思改进
- 适用于API模型（无法微调）

下一步：
1. 尝试更复杂的编程任务
2. 实现推理任务的Reflexion
3. 继续实践项目3：RAG记忆系统

═══════════════════════════════════════════════════════════════════
```

---

## 参考资源

```
参考资源
═══════════════════════════════════════════════════════════════════

论文：
- Reflexion: Language Agents with Verbal Reinforcement Learning (2023)
  https://arxiv.org/abs/2303.11366

代码：
- Reflexion官方实现
  https://github.com/noahshinn/reflexion

相关项目：
- 实践项目1：简单ReAct代理
- 实践项目3：RAG记忆系统
- 实践项目4：多代理协作

═══════════════════════════════════════════════════════════════════
```

---

*实践项目2完成于 2026-07-04*
*预计学习时间：3-4小时*
*难度：⭐⭐ 初级-中级*
