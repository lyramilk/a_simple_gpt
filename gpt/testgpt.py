# nohup python3 testgpt.py > train.log &

import json
import torch
import requests
import os
import gpt0
import time
import logging
logging.basicConfig(filename="testgpt.log",level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s",datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

"""
下载数据
wget https://www.modelscope.cn/datasets/gongjy/minimind_dataset/resolve/master/pretrain_hq.jsonl
wget https://www.modelscope.cn/datasets/gongjy/minimind_dataset/resolve/master/sft_512.jsonl
wget https://www.modelscope.cn/datasets/gongjy/minimind_dataset/resolve/master/sft_2048.jsonl
#wget https://www.modelscope.cn/datasets/gongjy/minimind_dataset/resolve/master/dpo.jsonl
"""

if torch.cuda.is_available():
	torch.set_default_device("cuda")
else:
	torch.set_default_device("cpu")
torch.set_default_dtype(torch.bfloat16);


dataset_path = "/data/coding/minimind"
model_path = "/data/coding/"



class WebDownloader:
	def __init__(self,dataset_path:str):
		self.dataset_path = dataset_path
		if not os.path.exists(self.dataset_path):
			os.makedirs(self.dataset_path)

	def load_data(self,url:str,filename:str):
		output_path = os.path.join(self.dataset_path,filename)
		# 支持断点续传
		if os.path.exists(output_path):
			print(f"文件{filename}已存在，跳过下载",flush=True)
			return output_path;
		MB = 1024*1024

		remote_file_size = 0;
		response = requests.head(url)
		if response.headers.get("Content-Length"):
			remote_file_size = int(response.headers.get("Content-Length"))

		remote_file_size_MB = remote_file_size / MB
		response = requests.get(url, stream=True)
	
		
		lasttime = time.time()

		downloaded_size = 0
		with open(output_path, "ab") as f:
			for chunk in response.iter_content(chunk_size=1024):
				if chunk:
					f.write(chunk)
					downloaded_size += len(chunk)
					if time.time() - lasttime > 1:
						cache_size = downloaded_size / MB
						if remote_file_size != 0:
							print(f"下载进度：{cache_size}/{remote_file_size_MB}MB，{cache_size/(remote_file_size_MB)*100}%，用时：{time.time() - lasttime}秒",flush=True)
						else:
							print(f"下载进度：{cache_size}MB",flush=True)
						lasttime = time.time()
		print(f"文件{filename}下载完成，大小为{downloaded_size/MB}MB",flush=True)
		return output_path



wd = WebDownloader(dataset_path)


def pt():
	with open(wd.load_data("https://www.modelscope.cn/datasets/gongjy/minimind_dataset/resolve/master/pretrain_hq.jsonl","pretrain_hq.jsonl"),"r") as f:
		for l in f:
			o = json.loads(l);
			sample = o.get("text","");
			yield sample;


def sft512():
	with open(wd.load_data("https://www.modelscope.cn/datasets/gongjy/minimind_dataset/resolve/master/sft_512.jsonl","sft_512.jsonl"),"r") as f:
		for l in f:
			o = json.loads(l);
			sample = "";
			for qa in o["conversations"]:
				sample += "<｜begin▁of▁sentence｜>" + qa.get("role","bot") + " " + qa.get("content","") + "<｜end▁of▁sentence｜>"
			yield sample

def sft2048():
	with open(wd.load_data("https://www.modelscope.cn/datasets/gongjy/minimind_dataset/resolve/master/sft_2048.jsonl","sft_2048.jsonl"),"r") as f:
		for l in f:
			o = json.loads(l);
			sample = "";
			for qa in o["conversations"]:
				sample += "<｜begin▁of▁sentence｜>" + qa.get("role","bot") + " " + qa.get("content","") + "<｜end▁of▁sentence｜>"
			yield sample


def dbo():
	dt = [];
	with open(wd.load_data("https://www.modelscope.cn/datasets/gongjy/minimind_dataset/resolve/master/dpo.jsonl","dpo.jsonl"),"r") as f:
		for l in f:
			o = json.loads(l);
			sample = "";
			chosen = o.get("chosen",{});
			rejected = o.get("rejected",{});
			sample += "<｜begin▁of▁sentence｜>" + chosen.get("role","bot") + " " + chosen.get("content","") + "<｜end▁of▁sentence｜>"
			sample += "<｜begin▁of▁sentence｜>" + rejected.get("role","bot") + " " + rejected.get("content","") + "<｜end▁of▁sentence｜>"
			yield sample





if not os.path.exists(model_path):
	os.makedirs(model_path)

pretrain_path = os.path.join(model_path,"pretrain.pth")
sft512_path = os.path.join(model_path,"sft512.pth")
sft2048_path = os.path.join(model_path,"sft2048.pth")
dpo_path = os.path.join(model_path,"dpo.pth")


tinylm = gpt0.TextGpt(d_model=512,num_attention_heads=8,num_layers=6,d_ff=1536,context_len=16384)

def test(input_text:str):
	for w in tinylm.generate(input_text, max_new_tokens=200, top_p=0.9, temperature=0.7):
		if w == "<｜end▁of▁sentence｜>":
			print();
			break;
		print(w, end="", flush=True)

# 预训练
if os.path.exists(pretrain_path):
	tinylm.load_model(pretrain_path)
else:
	time_start = time.time()
	tinylm.learn(pt(), epochs=1, learning_rate=1e-3, batch_size=20)
	print("预训练完成，用时：",int(time.time() - time_start),flush=True)
	tinylm.save_model(pretrain_path)

test("中国")

# 微调512
if os.path.exists(sft512_path):
	tinylm.load_model(sft512_path)
else:
	time_start = time.time()
	tinylm.learn(sft512(), epochs=1, learning_rate=1e-4, batch_size=10)
	print("微调512完成，用时：",int(time.time() - time_start),flush=True)
	tinylm.save_model(sft512_path)

test("<｜begin▁of▁sentence｜>user 中国的首都是哪里？<｜end▁of▁sentence｜><｜begin▁of▁sentence｜>assistant ")

# 微调
if os.path.exists(sft2048_path):
	tinylm.load_model(sft2048_path)
else:
	time_start = time.time()
	tinylm.learn(sft2048(), epochs=1, learning_rate=1e-4, batch_size=10)
	print("微调2048完成，用时：",int(time.time() - time_start),flush=True)
	tinylm.save_model(sft2048_path)

test("<｜begin▁of▁sentence｜>user 中国的首都是哪里？<｜end▁of▁sentence｜><｜begin▁of▁sentence｜>assistant ")

# 对抗训练
'''
if os.path.exists(dpo_path):
	tinylm.load_model(dpo_path)
else:
	time_start = time.time()
	data_count = 0
	total_bytes = 0
	bytes_per_second = 0
	sample_per_second = 0

	last_time = time.time()
	
	for dt in dbo():
		data_count += len(dt)
		batch_bytes = sum(len(d) for d in dt)
		total_bytes += batch_bytes
		bytes_per_second += batch_bytes
		sample_per_second += len(dt)
		loss = tinylm.learn(dt, epochs=1, learning_rate=1e-4, batch_size=2)
		1e-4, batch_size=10)
		cur_time = time.time()
		if cur_time - last_time > 1:
			print("对抗训练样本数：",data_count,",字节数：",total_bytes,",损失值：%.5f"%loss,",用时：",int(time.time() - time_start),",每秒字节数：%d"%(bytes_per_second/(cur_time - last_time)),",每秒样本数：%d"%(sample_per_second/(cur_time - last_time)),flush=True)
			sample_per_second = 0
			bytes_per_second = 0
			last_time = time.time()
	tinylm.save_model(dpo_path)
'''


