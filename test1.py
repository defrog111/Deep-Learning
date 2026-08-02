import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
from transfomers import 
# 1. 加载 tokenizer 和模型
model_name = "Qwen/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

model.eval()


# 2. 两句话长度不同
texts = [
    "I love machine learning.",
    "Hello."
]


# 3. padding=True：
#    把较短的句子补到和最长句子一样长
#
# return_tensors="pt"：
#    返回 PyTorch Tensor
inputs = tokenizer(
    texts,
    padding=True,
    return_tensors="pt"
)


# 4. 查看 tokenizer 的 padding token
print("pad_token:", tokenizer.pad_token)
print("pad_token_id:", tokenizer.pad_token_id)

print("\ninput_ids:")
print(inputs["input_ids"])

print("\nattention_mask:")
print(inputs["attention_mask"])


# 5. 把 token id 转回 token，观察 padding 出现在哪里
print("\nTokens:")

for i, token_ids in enumerate(inputs["input_ids"]):
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    print(f"\nSentence {i}:")
    print(tokens)


# 6. 输入模型
with torch.no_grad():
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )


# 7. 模型输出 logits
print("\nlogits shape:")
print(outputs.logits.shape)