from plugins.common import settings


def chat_init(history):
    history_formatted = None
    if history is not None:
        history_formatted = []
        tmp = []
        for i, old_chat in enumerate(history):
            if len(tmp) == 0 and old_chat['role'] == "user":
                tmp.append(old_chat['content'])
            elif old_chat['role'] == "AI" or old_chat['role'] == 'assistant':
                tmp.append(old_chat['content'])
                history_formatted.append(tuple(tmp))
                tmp = []
            else:
                continue
    return history_formatted


def chat_one(prompt, history_formatted, max_length, top_p, temperature, data):
    yield str(len(prompt)) + '字正在计算'
    response, history = model.stream_chat(tokenizer, prompt, temperature=temperature, top_p=top_p, max_new_tokens=max_length)
    yield response


def load_model():
    global model, tokenizer
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        settings.llm.path, trust_remote_code=True, revision="v0.1.0")
    model = AutoModelForCausalLM.from_pretrained(
        settings.llm.path, trust_remote_code=True, revision="v0.1.0").to(torch.bfloat16).cuda()
    model.eval()
