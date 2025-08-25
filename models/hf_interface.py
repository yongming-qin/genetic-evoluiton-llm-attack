"""
Hugging Face模型接口类
提供与GPT-OSS-120B模型交互的统一接口
"""
import os
import time
from datetime import datetime
from typing import Dict, List, Optional
from openai import OpenAI

from model_config import ModelConfig

class HFModelInterface:
    """Hugging Face模型接口类"""
    
    def __init__(self, 
                 hf_token: Optional[str] = None, 
                 hf_model_id: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model_name: Optional[str] = None):
        """
        初始化模型接口
        
        Args:
            hf_token: Hugging Face token
            hf_model_id: 模型ID
            base_url: API基础URL
            model_name: 模型名称，用于获取特定的token
        """
        self.model_name = model_name
        self.hf_token = hf_token or ModelConfig.get_hf_token(model_name)
        self.hf_model_id = hf_model_id or ModelConfig.get_target_model_config()['model_id']
        self.base_url = base_url or ModelConfig.get_target_model_config()['base_url']
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.hf_token
        )
        self.response_history: List[Dict] = []
    
    def generate(self, 
                 prompt: str,
                 system_prompt: str = "You are a helpful, harmless, and honest AI assistant.",
                 temperature: float = ModelConfig.get_target_model_config()['temperature'],
                 max_tokens: int = ModelConfig.get_target_model_config()['max_tokens'],
                 repetition_count: int = 1,
                 padding_tokens: int = 0,
                 language: str = 'en',
                 **kwargs) -> Dict:
        """
        生成模型响应
        
        Args:
            prompt: 用户输入提示
            system_prompt: 系统提示
            temperature: 温度参数
            max_tokens: 最大token数
            repetition_count: 重复次数
            padding_tokens: 填充token数
            language: 语言
            
        Returns:
            包含响应的字典
        """
        # 基础增强
        user_msg = prompt
        if repetition_count > 1:
            user_msg = (user_msg + " ") * repetition_count
        if padding_tokens > 0:
            user_msg = "PAD " * padding_tokens + user_msg

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ]

        try:
            completion = self.client.chat.completions.create(
                model=self.hf_model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            reply = completion.choices[0].message.content
        except Exception as e:
            reply = f"[HF router error] {e}"

        result = {"response": reply}
        self._log_interaction(user_msg, reply)
        return result
    
    def _log_interaction(self, prompt: str, response: str):
        """记录交互历史"""
        self.response_history.append({
            "prompt": prompt,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_history(self) -> List[Dict]:
        """获取交互历史"""
        return self.response_history.copy()
    
    def generate_with_reasoning(self, 
                               prompt: str,
                               system_prompt: str = "You are a helpful, harmless, and honest AI assistant.",
                               temperature: float = ModelConfig.get_target_model_config()['temperature'],
                               max_tokens: int = ModelConfig.get_target_model_config()['max_tokens'],
                               **kwargs) -> Dict:
        """
        生成模型响应，并尝试获取reasoning过程
        
        Args:
            prompt: 用户输入提示
            system_prompt: 系统提示
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            包含响应和reasoning过程的字典
        """
        # 构建包含reasoning指令的prompt - 强制要求展示思考过程
        reasoning_prompt = f"""IMPORTANT: You MUST show your step-by-step thinking process before answering.

{prompt}

REQUIRED: Let me think about this step by step:"""
        
        # 修改系统提示，强制要求展示思考过程
        enhanced_system_prompt = f"{system_prompt}\n\nCRITICAL: You must always show your detailed thinking process step by step before providing any answer. This is mandatory."
        
        messages = [
            {"role": "system", "content": enhanced_system_prompt},
            {"role": "user", "content": reasoning_prompt}
        ]

        try:
            # 检查是否是GPT-OSS-20B模型，如果是则使用streaming获取reasoning
            if "gpt-oss-20b" in self.hf_model_id.lower():
                return self._generate_with_streaming_reasoning(messages, temperature, max_tokens, **kwargs)
            else:
                # 其他模型使用普通方式
                completion = self.client.chat.completions.create(
                    model=self.hf_model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
                reply = completion.choices[0].message.content
                
                # 直接使用完整响应，reasoning已经通过prompt获取
                result = {
                    "response": reply,  # 完整响应
                    "reasoning": reply,  # 完整响应作为reasoning
                    "final_answer": ""   # 不分离final_answer
                }
                
                self._log_interaction(reasoning_prompt, reply)
                return result
            
        except Exception as e:
            reply = f"[HF router error] {e}"
            result = {
                "response": reply,
                "reasoning": reply,
                "final_answer": ""
            }
            
            self._log_interaction(reasoning_prompt, reply)
            return result
    

    
    def _generate_with_streaming_reasoning(self, messages: List[Dict], temperature: float, max_tokens: int, **kwargs) -> Dict:
        """
        使用streaming方式获取GPT-OSS-20B的reasoning过程
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            包含响应和reasoning过程的字典
        """
        try:
            # 使用streaming方式调用GPT-OSS-20B
            stream = self.client.chat.completions.create(
                model=self.hf_model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,  # 启用streaming
                **kwargs
            )
            
            reasoning_parts, final_parts = [], []
            
            # 处理streaming响应
            for chunk in stream:
                choice = chunk.choices[0].delta
                
                # 获取reasoning内容
                if hasattr(choice, "reasoning_content") and choice.reasoning_content:
                    reasoning_parts.append(choice.reasoning_content)
                
                # 获取最终答案内容
                if hasattr(choice, "content") and choice.content:
                    final_parts.append(choice.content)
            
            # 组合完整内容
            reasoning_text = "".join(reasoning_parts)
            final_answer_text = "".join(final_parts)
            full_response = reasoning_text + final_answer_text
            
            # 记录交互
            self._log_interaction(messages[-1]["content"], full_response)
            
            return {
                "response": full_response,
                "reasoning": reasoning_text,
                "final_answer": final_answer_text
            }
            
        except Exception as e:
            print(f"Streaming reasoning generation failed: {e}")
            # 回退到普通方式
            return self._generate_with_streaming_reasoning_fallback(messages, temperature, max_tokens, **kwargs)
    
    def _generate_with_streaming_reasoning_fallback(self, messages: List[Dict], temperature: float, max_tokens: int, **kwargs) -> Dict:
        """
        Streaming失败时的回退方法
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.hf_model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,  # 禁用streaming
                **kwargs
            )
            reply = completion.choices[0].message.content
            
            # 直接使用完整响应
            return {
                "response": reply,
                "reasoning": reply,
                "final_answer": ""
            }
            
        except Exception as e:
            return {
                "response": f"[HF router error] {e}",
                "reasoning": f"[HF router error] {e}",
                "final_answer": ""
            }
    
    def clear_history(self):
        """清空交互历史"""
        self.response_history.clear()
