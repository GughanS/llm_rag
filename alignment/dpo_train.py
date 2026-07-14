import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from copy import deepcopy

class DPODataset(Dataset):
    def __init__(self, jsonl_path, tokenizer):
        self.data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def _tokenize_pair(self, prompt, response):
        # Format: "User: {prompt}\nAssistant: {response}<|endoftext|>"
        full_text = f"User: {prompt}\nAssistant: {response}{self.tokenizer.eos_token}"
        prompt_text = f"User: {prompt}\nAssistant: "
        
        # We need the input ids and a mask for which tokens belong to the response
        encoded_full = self.tokenizer(full_text, return_tensors="pt")
        encoded_prompt = self.tokenizer(prompt_text, return_tensors="pt")
        
        input_ids = encoded_full["input_ids"][0]
        prompt_len = encoded_prompt["input_ids"].shape[1]
        
        # Mask is 0 for prompt, 1 for response
        mask = torch.zeros_like(input_ids)
        mask[prompt_len:] = 1
        
        return input_ids, mask

    def __getitem__(self, idx):
        item = self.data[idx]
        chosen_ids, chosen_mask = self._tokenize_pair(item["prompt"], item["chosen"])
        rejected_ids, rejected_mask = self._tokenize_pair(item["prompt"], item["rejected"])
        
        return {
            "chosen_ids": chosen_ids,
            "chosen_mask": chosen_mask,
            "rejected_ids": rejected_ids,
            "rejected_mask": rejected_mask
        }

def collate_fn(batch, pad_token_id):
    # Pad sequences to max length in batch
    def pad_tensors(tensors, pad_val):
        max_len = max(t.shape[0] for t in tensors)
        padded = []
        for t in tensors:
            pad_len = max_len - t.shape[0]
            padded.append(F.pad(t, (0, pad_len), value=pad_val))
        return torch.stack(padded)

    chosen_ids = pad_tensors([b["chosen_ids"] for b in batch], pad_token_id)
    chosen_mask = pad_tensors([b["chosen_mask"] for b in batch], 0)
    rejected_ids = pad_tensors([b["rejected_ids"] for b in batch], pad_token_id)
    rejected_mask = pad_tensors([b["rejected_mask"] for b in batch], 0)
    
    return {
        "chosen_ids": chosen_ids,
        "chosen_mask": chosen_mask,
        "rejected_ids": rejected_ids,
        "rejected_mask": rejected_mask
    }

def get_batch_logps(model, input_ids, mask):
    """
    Computes the log probabilities of the response tokens for a given model.
    """
    # Forward pass
    outputs = model(input_ids)
    logits = outputs.logits  # (batch, seq_len, vocab_size)
    
    # Shift logits and labels (predict next token)
    # Logits from 0 to N-1 predict tokens 1 to N
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = mask[:, 1:].contiguous()
    
    # Compute log probabilities of the actual chosen/rejected tokens
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # Gather the log probabilities of the target tokens
    # gather shape: (batch, seq_len-1, 1) -> squeeze to (batch, seq_len-1)
    token_logps = torch.gather(log_probs, dim=2, index=shift_labels.unsqueeze(2)).squeeze(-1)
    
    # Mask out the prompt tokens (only sum over response tokens)
    response_logps = (token_logps * shift_mask).sum(dim=1)
    return response_logps

def train_dpo():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Initializing {model_name} with random weights (bypassing HF block)...")
    config = AutoConfig.from_pretrained(model_name)
    active_model = AutoModelForCausalLM.from_config(config).to(device)
        
    # Reference model is a frozen copy of the pre-DPO model
    ref_model = deepcopy(active_model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
        
    active_model.train()
    
    dataset = DPODataset("data/dpo_train.jsonl", tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, 
                            collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id))
    
    optimizer = torch.optim.AdamW(active_model.parameters(), lr=1e-5)
    
    beta = 0.1 # DPO temperature
    epochs = 3
    
    print("Starting DPO training...")
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            chosen_ids = batch["chosen_ids"].to(device)
            chosen_mask = batch["chosen_mask"].to(device)
            rejected_ids = batch["rejected_ids"].to(device)
            rejected_mask = batch["rejected_mask"].to(device)
            
            # 1. Forward passes on Active Model
            pi_logps_chosen = get_batch_logps(active_model, chosen_ids, chosen_mask)
            pi_logps_rejected = get_batch_logps(active_model, rejected_ids, rejected_mask)
            
            # 2. Forward passes on Reference Model (no gradients)
            with torch.no_grad():
                ref_logps_chosen = get_batch_logps(ref_model, chosen_ids, chosen_mask)
                ref_logps_rejected = get_batch_logps(ref_model, rejected_ids, rejected_mask)
                
            # 3. Compute DPO Loss
            pi_ratio = pi_logps_chosen - pi_logps_rejected
            ref_ratio = ref_logps_chosen - ref_logps_rejected
            
            # Custom DPO Loss Formula: -log(sigmoid(beta * (pi_ratio - ref_ratio)))
            logits = beta * (pi_ratio - ref_ratio)
            loss = -F.logsigmoid(logits).mean()
            
            # 4. Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")
        
    # Save the aligned model
    active_model.save_pretrained("dpo_aligned_model")
    tokenizer.save_pretrained("dpo_aligned_model")
    print("DPO Alignment complete. Saved to dpo_aligned_model/")

if __name__ == "__main__":
    train_dpo()
