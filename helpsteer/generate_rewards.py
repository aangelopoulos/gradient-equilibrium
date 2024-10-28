import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm

device = 'cuda:0' if torch.cuda.is_available() else "cpu"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('Ray2333/GRM-gemma2-2B-rewardmodel-ft')

print("Loading reward model...")
reward_model = AutoModelForSequenceClassification.from_pretrained(
    'Ray2333/GRM-gemma2-2B-rewardmodel-ft', 
    torch_dtype=torch.float16,
    device_map=device,
)

# Set max_length to a reasonable value (e.g., 2048 for most transformer models)
max_length = 2048

print("Loading dataset...")
data = load_dataset("nvidia/HelpSteer2", split='validation')

print("Making iterable...")
data_iter = data.to_iterable_dataset()

reward_model.eval()

kwargs = {
    "padding": True,
    "truncation": True,
    "max_length": max_length,
    "return_tensors": "pt"
}

rewards = []

print("Loop...")
with torch.no_grad():
    for row in tqdm(data_iter):
        prompt = row['prompt']
        response = row['response']
        gt = row['helpfulness']
        
        message = [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': response}
        ]
        message_template = tokenizer.apply_chat_template(message, tokenize=False)
        
        try:
            tokens = tokenizer.encode_plus(message_template, **kwargs)
            
            reward_tensor = reward_model(
                tokens["input_ids"].to(device),
                attention_mask=tokens["attention_mask"].to(device)
            )[0]
            
            reward = reward_tensor.cpu().detach().item()
            rewards.append(reward)
            
        except Exception as e:
            print(f"Error processing row: {e}")
            rewards.append(None)  # or some default value
            continue

df = data.to_pandas()
df['rm_rewards'] = rewards

# Remove any None values before saving
df = df.dropna(subset=['rm_rewards'])

df.to_json("rewards.json", orient='records', indent=1)