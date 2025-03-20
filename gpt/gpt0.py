import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Generator, Iterable
import time
from torch.nn.utils.rnn import pad_sequence
from modelscope import snapshot_download
from transformers import AutoTokenizer
import logging
from multiprocessing import Pool,Queue
import multiprocessing as mp




class Embedding(nn.Module):
    """
    嵌入层
    """
    def __init__(self, vocab_size: int, d_model: int):
        """
        初始化嵌入层
        Args:
            vocab_size: 词汇表大小
            d_model: 模型维度
        """
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入张量
        Returns:
            torch.Tensor: 嵌入后的张量
        """
        return self.embedding(x) * math.sqrt(self.d_model)

class Normalization(nn.Module):
    """
    归一化层
    """
    def __init__(self, d_model: int):
        """
        初始化归一化层
        Args:
            d_model: 模型维度
        """
        super().__init__()
        self.k = nn.Parameter(torch.ones(d_model))
        self.b = nn.Parameter(torch.zeros(d_model))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            input: 输入张量 (batch_size, seq_len, d_model)
        Returns:
            torch.Tensor: 归一化后的张量 (batch_size, seq_len, d_model)
        """
        mean = input.mean(dim=-1, keepdim=True)
        std = input.std(dim=-1, keepdim=True)
        x = (input - mean) / (std + 1e-5)
        # 避免使用原地操作，而是创建新的张量
        return self.k * x + self.b


class PositionalEncoding(nn.Module):
    """
    位置编码层
    """
    def __init__(self, d_model: int, context_len: int):
        """
        初始化位置编码层
        Args:
            d_model: 模型维度
            context_len: 上下文长度
        """
        super().__init__()
        self.positional_encoding = nn.Parameter(torch.zeros(context_len, d_model)) # (context_len, d_model)
        self.positional_encoding.requires_grad = False
        
        position = torch.arange(0, context_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.positional_encoding.data[:, 0::2] = torch.sin(position * div_term)
        self.positional_encoding.data[:, 1::2] = torch.cos(position * div_term)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            input: 输入张量 (batch_size, seq_len, d_model)
        Returns:
            torch.Tensor: 位置编码后的张量 (batch_size, seq_len, d_model)
        """
        seq_len = input.size(1)
        return input + self.positional_encoding[:seq_len].unsqueeze(0)

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    """
    def __init__(self, d_model: int, num_attention_heads: int):
        """
        初始化多头注意力机制
        Args:
            d_model: 模型维度
            num_attention_heads: 注意力头数
        """
        super().__init__()
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.head_dim = d_model // num_attention_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.dk_sqrt = math.sqrt(self.head_dim)

    def scaled_dot_product_attention(self, input_q: torch.Tensor, input_k: torch.Tensor, input_v: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        缩放点积注意力
        Args:
            input_q: 查询张量 (batch_size, num_attention_heads, seq_len, head_dim)
            input_k: 键张量 (batch_size, num_attention_heads, seq_len, head_dim)
            input_v: 值张量 (batch_size, num_attention_heads, seq_len, head_dim)
            mask: 掩码张量 (batch_size, num_attention_heads, seq_len, seq_len)
        Returns:
            torch.Tensor: 注意力输出张量 (batch_size, num_attention_heads, seq_len, head_dim)
        """
        # 交换k的最后两个维度，以便进行矩阵乘法，否则形状不符合矩阵乘法要求。
        kt = input_k.transpose(2, 3) # kt (batch_size, num_attention_heads, head_dim, seq_len)
        qk = torch.matmul(input_q, kt) # qk (batch_size, num_attention_heads, seq_len, seq_len)
        # dk = input_k.size(-1)  dk和dk_sqrt由于在初始化的时候已经确定，所以不需要再计算
        
        # 缩放点积注意力
        scaled_qk = qk / self.dk_sqrt
        
        # 应用掩码
        if mask is not None:
            scaled_qk = scaled_qk.masked_fill(mask == 0, float('-inf'))
        
        # 计算注意力权重
        attn = F.softmax(scaled_qk, dim=-1) # (batch_size, num_attention_heads, seq_len, seq_len)
        
        # 计算输出
        out = torch.matmul(attn, input_v) # (batch_size, num_attention_heads, seq_len, head_dim)
        return out

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        前向传播
        Args:
            input: 输入张量 (batch_size, seq_len, d_model)
            mask: 掩码张量 (batch_size, num_attention_heads, seq_len, seq_len)
        Returns:
            torch.Tensor: 注意力输出张量 (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = input.size()
        
        input_q = self.wq(input) # 
        input_k = self.wk(input)
        input_v = self.wv(input)

        input_q = input_q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2) # (batch_size, num_attention_heads, seq_len, head_dim)
        input_k = input_k.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2) # (batch_size, num_attention_heads, seq_len, head_dim) 
        input_v = input_v.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2) # (batch_size, num_attention_heads, seq_len, head_dim)
            
        output = self.scaled_dot_product_attention(input_q, input_k, input_v, mask) # (batch_size, num_attention_heads, seq_len, head_dim)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model) # (batch_size, seq_len, d_model)
        output = self.wo(output) # (batch_size, seq_len, d_model)
        return output # (batch_size, seq_len, d_model)


class FeedForward(nn.Module):
    """
    前馈神经网络
    """
    def __init__(self, d_model: int, d_ff: int):
        """
        初始化前馈神经网络
        Args:
            d_model: 模型维度
            d_ff: 前馈网络维度
        """
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            input: 输入张量 (batch_size, seq_len, d_model)
        Returns:
            torch.Tensor: 前馈神经网络输出张量 (batch_size, seq_len, d_model)
        """
        return self.w2(F.silu(self.w1(input)))


class Block(nn.Module):
    """
    注意力和前馈神经网络合并成一个块一起翻倍
    """
    def __init__(self, d_model: int, num_attention_heads: int, d_ff: int):
        """
        初始化块
        Args:
            d_model: 模型维度
            num_attention_heads: 注意力头数
            d_ff: 前馈网络维度
        """
        super().__init__()
        self.norm1 = Normalization(d_model)
        self.mha = MultiHeadAttention(d_model, num_attention_heads)
        self.norm2 = Normalization(d_model)
        self.ff = FeedForward(d_model, d_ff)

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        前向传播
        Args:
            input: 输入张量 (batch_size, seq_len, d_model)
            mask: 掩码张量 (batch_size, num_attention_heads, seq_len, seq_len)
        Returns:
            torch.Tensor: 输出张量 (batch_size, seq_len, d_model)
        """
        # 避免使用原地操作（+=），而是创建新的张量
        # pre_norm 模式比 post_norm 模式更稳定
        attn_output = input + self.mha(self.norm1(input), mask) # (batch_size, seq_len, d_model)
        output = attn_output + self.ff(self.norm2(attn_output)) # (batch_size, seq_len, d_model)
        return output

class OutputProbability(nn.Module):
    """
    输出概率线性层
    """
    def __init__(self, d_model: int, vocab_size: int):
        """
        初始化输出概率线性层
        Args:
            d_model: 模型维度
            vocab_size: 词汇表大小
        """
        super().__init__()
        self.norm = Normalization(d_model) # (batch_size, seq_len, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size) # (batch_size, seq_len, vocab_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            input: 输入张量 (batch_size, seq_len, d_model)
        Returns:
            torch.Tensor: 输出张量 (batch_size, seq_len, vocab_size)
        """
        x = self.norm(input) # (batch_size, seq_len, d_model)
        return self.lm_head(x) # (batch_size, seq_len, vocab_size)


class Gpt(nn.Module):
    """
    生成式预训练语言模型
    """
    def __init__(self, vocab_size: int, d_model: int = 768, num_attention_heads: int = 8, num_layers: int = 6, d_ff: int = 2048, context_len: int = 1024,end_token_id: int = 0):
        super().__init__()
        """
        初始化
        Args:
            vocab_size: 词汇表大小
            d_model: 模型维度
            num_attention_heads: 注意力头数，注意d_model要能被num_attention_heads整除
            num_layers: 层数，通常设置为6~12
            d_ff: 前馈网络维度，通常设置为d_model的2.5~8倍
            context_len: 上下文长度，通常设置为1024~2048
        """
        self.context_len = context_len
        self.embedding = Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, context_len=context_len)
        self.num_attention_heads = num_attention_heads
        self.blocks = nn.ModuleList([Block(d_model, num_attention_heads, d_ff) for _ in range(num_layers)])
        self.output_probability = OutputProbability(d_model, vocab_size)
        self.end_token_id = end_token_id

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            input: 输入张量 (batch_size, seq_len)
        Returns:
            torch.Tensor: 输出张量 (batch_size, seq_len, vocab_size)
        """
        batch_size = input.size(0)
        input = self.embedding(input) # (batch_size, seq_len, d_model)
        input = self.positional_encoding(input) # (batch_size, seq_len, d_model)
        
        # 创建因果掩码，确保模型只能看到当前位置之前的标记
        seq_len = input.size(1)
        # 创建下三角矩阵作为掩码
        mask = torch.tril(torch.ones((seq_len, seq_len), device=input.device))
        mask = mask.unsqueeze(0).expand(batch_size,self.num_attention_heads, -1, -1)
        
        for block in self.blocks:
            input = block(input, mask) # (batch_size, seq_len, d_model)
        
        logits = self.output_probability(input) # (batch_size, seq_len, vocab_size)
        return logits

    @torch.inference_mode()
    def generate_word_stream(self, input: torch.Tensor, max_new_tokens: int = 1, top_k: Optional[int]=None, top_p: Optional[float]=None, temperature: Optional[float]=None) -> Generator[int, None, None]:
        """
        生成单词流
        Args:
            input: 输入张量 (batch_size, seq_len)
            max_new_tokens: 最多生成的新token数量
            top_k: 如果>0，只保留概率最高的top_k个token
            top_p: 如果>0，只保留累积概率达到top_p的token
            temperature: 温度参数，控制采样的随机性
        Yields:
            int: 输出token
        """
        # 确保输入是二维的，形状为(1, seq_len)
        if input.dim() == 1:
            input = input.unsqueeze(0)  # 添加批次维度
        else:
            if input.size(0) > 1:
                input = input[0].unsqueeze(0)  # 只使用第一个批次
            # 如果已经是正确的形状(1, seq_len)，则不需要修改
            
        batch_size = input.size(0)
        current_seq = input

        yield from current_seq[0];

        
        for _ in range(max_new_tokens):
            # 如果序列长度超过最大长度，截断序列
            if current_seq.size(1) >= self.context_len:
                current_seq = current_seq[:, -self.context_len:]
                
            # 对当前序列进行前向传播
            current_input = self.embedding(current_seq)  # (batch_size, seq_len, d_model)
            current_input = self.positional_encoding(current_input)  # (batch_size, seq_len, d_model)
            
            # 创建因果掩码
            seq_len = current_input.size(1)
            mask = torch.tril(torch.ones((seq_len, seq_len), device=current_input.device))
            mask = mask.unsqueeze(0).expand(batch_size,self.num_attention_heads, -1, -1)
            
            # 通过Transformer块
            for block in self.blocks:
                current_input = block(current_input, mask)  # (batch_size, seq_len, d_model)
                
            # 获取logits
            logits = self.output_probability(current_input)  # (batch_size, seq_len, vocab_size)
            
            # 只取最后一个时间步的logits用于生成下一个token
            last_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
            
            # 生成下一个token
            next_token = self.logits_to_token(last_token_logits, top_k, top_p, temperature)  # (batch_size, 1)

            # 判断next_token中所有token是否都为结束标记
            if (next_token == self.end_token_id).all():
                break
            
            yield next_token[0].item()
            # 将新token添加到序列中
            current_seq = torch.cat([current_seq, next_token], dim=1)
            
        yield self.end_token_id;


    @torch.inference_mode()
    def generate_batch(self, input: torch.Tensor, max_new_tokens: int = 1, top_k: int=None, top_p: float=None, temperature: float=None) -> torch.Tensor:
        """
        生成新的token序列
        Args:
            input: 输入序列，形状为(batch_size, seq_len)或(seq_len)
            max_new_tokens: 最多生成的新token数量
            top_k: 如果>0，只保留概率最高的top_k个token
            top_p: 如果>0，只保留累积概率达到top_p的token
            temperature: 温度参数，控制采样的随机性
        Returns:
            生成的完整序列，包括输入序列
        """
        # 确保输入是二维的，形状为(batch_size, seq_len)
        if input.dim() == 1:
            input = input.unsqueeze(0)  # 添加批次维度
            
        batch_size = input.size(0)
        current_seq = input.clone()
        
        for _ in range(max_new_tokens):
            # 如果序列长度超过最大长度，截断序列
            if current_seq.size(1) >= self.context_len:
                current_seq = current_seq[:, -self.context_len:]
                
            # 对当前序列进行前向传播
            current_input = self.embedding(current_seq)  # (batch_size, seq_len, d_model)
            current_input = self.positional_encoding(current_input)  # (batch_size, seq_len, d_model)
            
            # 创建因果掩码
            seq_len = current_input.size(1)
            mask = torch.tril(torch.ones((seq_len, seq_len), device=current_input.device))
            mask = mask.unsqueeze(0).expand(batch_size,self.num_attention_heads, -1, -1)
            
            # 通过Transformer块
            for block in self.blocks:
                current_input = block(current_input, mask)  # (batch_size, seq_len, d_model)
                
            # 获取logits
            logits = self.output_probability(current_input)  # (batch_size, seq_len, vocab_size)
            
            # 只取最后一个时间步的logits用于生成下一个token
            last_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
            
            # 生成下一个token
            next_token = self.logits_to_token(last_token_logits, top_k, top_p, temperature)  # (batch_size, 1)

            # 判断next_token中所有token是否都为结束标记
            if (next_token == self.end_token_id).all():
                break
            
            # 将新token添加到序列中
            current_seq = torch.cat([current_seq, next_token], dim=1)
            
        return current_seq  # 返回完整序列，包括输入序列

    def top_p(self, logits_i: torch.Tensor, top_p: float=0.0) -> torch.Tensor:
        """
        根据top_p进行采样
        Args:
            logits_i: 输入张量 (vocab_size)
            top_p: 如果>0，只保留累积概率达到top_p的token
        Returns:
            torch.Tensor: 输出张量 (vocab_size)
        """
        sorted_logits, sorted_indices = torch.sort(logits_i, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p;
        sorted_indices_to_remove[0] = False
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        return indices_to_remove


    def logits_to_token(self, logits: torch.Tensor, top_k: Optional[int]=None, top_p: Optional[float]=None, temperature: Optional[float]=None) -> torch.Tensor:
        """
        根据top_k和top_p进行采样
        Args:
            logits: 输入张量 (batch_size, vocab_size)
            top_k: 如果>0，只保留概率最高的top_k个token
            top_p: 如果>0，只保留累积概率达到top_p的token
            temperature: 温度参数，控制采样的随机性，温度越高，采样越随机
        Returns:
            torch.Tensor: 输出张量 (batch_size, vocab_size)
        """
        
        # 如果temperature有效，则对logits进行维度缩放
        if temperature is not None:
            if temperature <= 0.0:
                return torch.argmax(logits, dim=-1)
            # 温度越高，logits各元素越接近，概率越均匀。温度越低，logits各元素差异越大，概率越集中。
            logits = logits / temperature
        else:
            if (top_k is None or top_k <= 0) and top_p is None:
                return torch.argmax(logits, dim=-1)

        # 如果top_k有效，则只保留概率最高的top_k个token
        if top_k is not None and top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        if top_p is not None:
            if top_p > 0.0:
                for i in range(logits.size(0)):
                    indices_to_remove = self.top_p(logits[i], top_p)
                    logits[i, indices_to_remove] = float('-inf')
            else:
                return torch.argmax(logits, dim=-1)

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token

    def back_propagate(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        反向传播
        Args:
            input: 输入张量 (batch_size, seq_len)
            target: 目标张量 (batch_size, seq_len)
        Returns:
            torch.Tensor: 损失值
        """
        logits = self.forward(input) # (batch_size, seq_len, vocab_size)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1),ignore_index=self.end_token_id)
        return loss
    
    def learn(self, samples: Iterable[torch.Tensor], epochs: int, stop_loss: float = 0.0, learning_rate: float = 1e-4, batch_size: int = 16) -> float:
        """
        训练
        Args:
            samples: 样本迭代器
            epochs: 训练轮数
            stop_loss: 停止训练的损失值
            learning_rate: 学习率
            batch_size: 批量大小，一次训练多少条样本
        """
        
        self.train() # 设置为训练模式
        
        # 将训练数据转换为输入和目标
        stime = time.time()
        lasttime = time.time();

        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        
        step = 1
        total_samples = 0
        last_report_samples = 0
        loss_item = 0.0
        skipped_batches = 0  # 记录跳过的批次数量
        
        for epoch in range(epochs):
            for batch in samples:
                try:
                    batch_input = [];
                    batch_target = [];
                    for sample in batch:
                        one_input = sample[:-1]
                        one_target = sample[1:]
                        batch_input.append(one_input)
                        batch_target.append(one_target)
                        
                    batch_input = pad_sequence(batch_input, batch_first=True, padding_value=self.end_token_id)
                    batch_target = pad_sequence(batch_target, batch_first=True, padding_value=self.end_token_id)

                    loss = self.back_propagate(batch_input, batch_target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_item = loss.item()
                    total_samples += batch_input.size(0)
                    if step % 100 == 0:
                        tm = time.time();
                        time_diff = tm - lasttime  # 上次汇报时间到现在经过的时间
                        samples_diff = total_samples - last_report_samples  # 上次汇报时间到现在又处理了多少样本
                        speed = samples_diff / time_diff
                        logging.info(f"训练轮数: {epoch}, 步数: {step}, 损失值: {loss_item}, 已处理样本数: {total_samples}，处理速度{speed:.2f}，跳过批次数: {skipped_batches}")
                        last_report_samples = total_samples;
                        lasttime = tm;
                    step += 1
                    del batch_input
                    del batch_target
                
                except torch.cuda.OutOfMemoryError:
                    # 捕获CUDA内存不足错误
                    skipped_batches += 1
                    logging.warning(f"CUDA内存不足，跳过当前批次。已跳过批次数: {skipped_batches}")
                    # 尝试清理内存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        if skipped_batches > 0:
            logging.warning(f"训练完成，总共跳过了 {skipped_batches} 个批次")
        
        return loss_item


_process_tokenizer = None

class TextBatchTokenizer(Iterable[torch.Tensor]):
    """
    文本分词器
    """
    def __init__(self, samples: Iterable[str], batch_size: int, tokenizer_path: str):
        self.samples = samples
        self.batch_size = batch_size
        self.tokenizer_path = tokenizer_path  # 分词器路径
        # 使用类方法作为初始化函数
        self.pool = Pool(10, initializer=self._initialize_worker, initargs=(self.tokenizer_path,))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pool:
            self.pool.close()
            self.pool.join()

    @classmethod
    def _initialize_worker(cls,tokenizer_path):
        """初始化工作进程，加载分词器到类变量中"""
        global _process_tokenizer
        _process_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print(f"进程 {mp.current_process().name} 初始化完成，分词器已加载")

    @classmethod
    def _encode_text(cls,sample):
        """使用类变量中的分词器进行编码"""
        global _process_tokenizer
        return _process_tokenizer.encode(sample, add_special_tokens=False)

    def __iter__(self) -> Iterable[torch.Tensor]:
        list_samples = []
        for sample in self.samples:
            list_samples.append(sample)
            if len(list_samples) >= self.batch_size:
                yield self.encode_batch(list_samples)
                list_samples = []

        if len(list_samples) > 0:
            yield self.encode_batch(list_samples)
            list_samples = []

    def encode_batch(self, list_samples: list[str]) -> list[torch.Tensor]:
        # 使用类方法进行编码
        results = self.pool.map(self._encode_text, list_samples)
        return [torch.tensor(result) for result in results]



class TextGpt:
    """
    文本生成工具
    """
    def __init__(self, d_model: int = 768, num_attention_heads: int = 64, num_layers: int = 6, d_ff: int = 2048, context_len: int = 1024,end_token_id: int = 0):
        """
        初始化
        Args:
            d_model: 模型维度
            num_attention_heads: 注意力头数
            num_layers: 层数
            d_ff: 前馈网络维度
            context_len: 上下文长度
            end_token_id: 结束token id
        """
        #tokenizer_path = snapshot_download("lyramilk/deepseek_v3_tokenizer",revision="v1.0.1")  # 下载分词器
        tokenizer_path = "/data/coding/minimind_tokenizer"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)  # 加载
        self.gpt = Gpt(self.tokenizer.vocab_size, d_model, num_attention_heads, num_layers, d_ff, context_len, end_token_id)

    def generate(self, input: str, max_new_tokens: int = 1, top_k: Optional[int]=None, top_p: Optional[float]=None, temperature: Optional[float]=None) -> Generator[str, None, None]:
        """
        生成文本
        Args:
            input: 输入文本
            max_new_tokens: 最多生成的新token数量
            top_k: 如果>0，只保留概率最高的top_k个token
            top_p: 如果>0，只保留累积概率达到top_p的token
            temperature: 温度参数，控制采样的随机性
        Yields:
            str: 输出文本
        """
        input_tokenstream = self.tokenizer.encode(input, add_special_tokens=False)
        input_tokenstream = torch.tensor(input_tokenstream)

        for output_token in self.gpt.generate_word_stream(input_tokenstream, max_new_tokens, top_k, top_p, temperature):
            if output_token == self.gpt.end_token_id:
                break;
            yield self.tokenizer.decode(output_token)


    def learn(self, samples: Iterable[str], epochs: int, stop_loss: float = 0.0, learning_rate: float = 1e-4, batch_size: int = 16) -> float:
        """
        训练
        Args:
            samples: 样本迭代器，每个样本是一个字符串
            epochs: 训练轮数
            stop_loss: 停止训练的损失值
            learning_rate: 学习率
            batch_size: 批量大小，一次训练多少条样本
        Returns:
            float: 平均损失值
        """
        
        tokenizer = TextTokenizer(samples, batch_size, self.tokenizer.name_or_path)

        
        #train_data = [tokenize(txt) for txt in samples]
        
        #train_data = [self.tokenizer.encode(txt, add_special_tokens=False) for txt in samples]
        #train_data = [torch.tensor(tokens) for tokens in train_data]
        return self.gpt.learn(tokenizer, epochs, stop_loss, learning_rate, batch_size)

    def parameter_count(self) -> int:
        """
        参数数量
        """
        return sum(p.numel() for p in self.gpt.parameters() if p.requires_grad)

    def save_model(self, path: str):
        """
        保存模型
        Args:
            path: 保存路径
        """
        torch.save(self.gpt.state_dict(), path)

    def load_model(self, path: str):
        """
        加载模型
        Args:
            path: 加载路径
        """
        self.gpt.load_state_dict(torch.load(path))

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    
    train_text = """
糖是面包的成分
蛋白质是面包的成分
"""

    train_data = [txt for txt in train_text.split("\n") if txt.strip()];


    for t in ("香蕉","橘子","苹果","梨","葡萄","西瓜","草莓","芒果"):
        y = ["糖","蛋白质"]
        
        train_data.append(f"{t}{y[0]}是{t}的成分") 
        train_data.append(f"{t}{y[1]}是{t}的成分")
        train_data.append(f"{t}的成分包含{t}{y[0]}和{t}{y[1]}")

    tinylm = TextGpt()

    print("训练样本")
    for txt in train_data:
        print(txt)
    tinylm.learn(train_data, epochs=10,stop_loss=0.01, learning_rate=0.001, batch_size=10)
    tinylm.save_model(r"e:\gpt.pth")
    #tinylm.load_model(r"e:\gpt.pth")

    test_text = "面包的成分包含";

    print("生成结果")
    for w in tinylm.generate(test_text, max_new_tokens=200, top_p=0.4, temperature=0.7):
        if w == "":
            print();
            break;
        print(w, end="", flush=True)
    for w in tinylm.generate(test_text, max_new_tokens=200, top_p=0.9, temperature=0.5):
        if w == "":
            print();
            break;
        print(w, end="", flush=True)
    for w in tinylm.generate(test_text, max_new_tokens=200, top_p=0.4, temperature=0.7):
        if w == "":
            print();
            break;
        print(w, end="", flush=True)
    for w in tinylm.generate(test_text, max_new_tokens=200, top_p=0.4, temperature=0.7):
        if w == "":
            print();
            break;
        print(w, end="", flush=True)
    for w in tinylm.generate(test_text, max_new_tokens=200, top_p=0.4, temperature=0.7):
        if w == "":
            print();
            break;
        print(w, end="", flush=True)
    for w in tinylm.generate(test_text, max_new_tokens=200, top_p=0.4, temperature=0.7):
        if w == "":
            print();
            break;
        print(w, end="", flush=True)
