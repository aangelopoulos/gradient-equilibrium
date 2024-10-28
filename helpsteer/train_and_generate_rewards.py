import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, Gemma2ForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
import wandb

# Configuration
config = {
    'model_name': 'Ray2333/GRM-gemma2-2B-rewardmodel-ft',
    'max_length': 1024,
    'batch_size': 2,
    'learning_rate': 2.0e-4,
    'num_epochs': 1,
    'warmup_steps': 0,
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'gradient_accumulation_steps': 1
}

# Initialize wandb for experiment tracking
wandb.init(project="reward-model-training", config=config)

def prepare_input(batch, tokenizer, config):
    messages = [
        [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': response}
        ]
        for prompt, response in zip(batch['prompt'], batch['response'])
    ]
    
    messages = [tokenizer.apply_chat_template(m, tokenize=False) for m in messages]
    
    inputs = tokenizer(
        messages,
        padding=True,
        truncation=True,
        max_length=config['max_length'],
        return_tensors="pt"
    )
    
    return inputs

def train_epoch(model, train_loader, tokenizer, optimizer, scheduler, config):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    progress_bar = tqdm(train_loader, desc="Training", total=len(train_loader))
    
    for step, batch in enumerate(progress_bar):
        inputs = prepare_input(batch, tokenizer, config)
        inputs = {k: v.to(config['device']) for k, v in inputs.items()}
        # Keep labels in float32 for better stability
        labels = torch.tensor(batch['helpfulness'], dtype=torch.float32).to(config['device'])

        
        outputs = model(
            input_ids=inputs['input_ids'].long(),
            attention_mask=inputs['attention_mask']
        )

        predictions = outputs['logits'].squeeze(-1)
        print(predictions)
        print(labels)
        loss = torch.nn.functional.mse_loss(predictions, labels)
        loss = loss / config['gradient_accumulation_steps']
        print(loss)
        loss.backward()
        
        if (step + 1) % config['gradient_accumulation_steps'] == 0 or (step + 1) == len(train_loader): # Data might not be divisable by batch size, so we optim step on last step.:
            # Clip gradients for stability
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * config['gradient_accumulation_steps']
        progress_bar.set_postfix({'loss': loss.item() * config['gradient_accumulation_steps']})
        
        wandb.log({
            'train_loss': loss.item() * config['gradient_accumulation_steps'],
            'learning_rate': scheduler.get_last_lr()[0]
        })
    
    return total_loss / len(train_loader)

def evaluate(model, eval_loader, tokenizer, config):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            inputs = prepare_input(batch, tokenizer, config)
            inputs = {k: v.to(config['device']) for k, v in inputs.items()}
            labels = batch['helpfulness']
            
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            
            predictions = outputs[0].squeeze().cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels)
    
        mse = F.mse_loss(torch.tensor(all_predictions), torch.tensor(all_labels))
    return mse, all_predictions

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model_name'],
        torch_dtype=torch.float32,
        device_map=config['device'],
    )
        
    print("Loading dataset...")
    train_data = load_dataset("nvidia/HelpSteer2", split='train')
    train_data = train_data.select(range(500))
    eval_data = load_dataset("nvidia/HelpSteer2", split='validation')

    train_loader = DataLoader(
        train_data,
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    eval_loader = DataLoader(
        eval_data,
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    # Optimizer and scheduler setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.0)
    num_training_steps = len(train_loader) * config['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=num_training_steps
    )
    
    # Training loop
    best_mse = float('inf')
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        avg_loss = train_epoch(model, train_loader, tokenizer, optimizer, scheduler, config)
        print(f"Average training loss: {avg_loss:.4f}")
        
        mse, predictions = evaluate(model, eval_loader, tokenizer, config)
        print(f"Validation MSE: {mse:.4f}")
        
        wandb.log({
            'epoch': epoch + 1,
            'avg_train_loss': avg_loss,
            'validation_mse': mse
        })
        
        if mse < best_mse:
            best_mse = mse
            #model.save_pretrained("best_model")
            print("Saved best model!")
    
    # Save final predictions
    eval_df = pd.DataFrame(eval_data)
    eval_df['rm_rewards'] = predictions
    eval_df.to_json("rewards.json", orient='records', indent=1)
    
    wandb.finish()

if __name__ == "__main__":
    main()