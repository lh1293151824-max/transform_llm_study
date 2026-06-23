from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
model_path = "./qwen-1.5b"
config = AutoConfig.from_pretrained(model_path)
#重新初始化的模型加载方式    model = AutoModelForCausalLM.from_config(config,trust_remote_code=True)
#使用官方训练好的权重的加载方式
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True)
print(f"model: {model}")
