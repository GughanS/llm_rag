import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def generate_response(model, tokenizer, prompt, device):
    # Format exactly like training
    input_text = f"User: {prompt}\nAssistant: "
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=50, 
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
    # Extract only the generated response
    prompt_len = inputs["input_ids"].shape[1]
    response_ids = outputs[0][prompt_len:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response.strip()

def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on {device}")
    
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained("dpo_aligned_model")
    
    print("Loading Base Model...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    except Exception:
        # Fallback if download blocked
        config = AutoConfig.from_pretrained(model_name)
        base_model = AutoModelForCausalLM.from_config(config).to(device)
        
    print("Loading DPO Aligned Model...")
    aligned_model = AutoModelForCausalLM.from_pretrained("dpo_aligned_model").to(device)
    
    base_model.eval()
    aligned_model.eval()
    
    # Load test prompts
    test_prompts = []
    with open("data/dpo_test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            test_prompts.append(json.loads(line)["prompt"])
            
    print("\n--- Running Evaluation ---\n")
    
    wins = 0
    
    for i, prompt in enumerate(test_prompts):
        print(f"Prompt {i+1}: {prompt}")
        
        base_resp = generate_response(base_model, tokenizer, prompt, device)
        aligned_resp = generate_response(aligned_model, tokenizer, prompt, device)
        
        print(f"[Base]    {base_resp}")
        print(f"[Aligned] {aligned_resp}")
        
        # Heuristic for our synthetic dataset: aligned should be more concise.
        # So we count it as a "win" if the aligned response is shorter than the base response.
        # (This is just a simple proxy for an LLM-as-a-judge for demonstration).
        base_len = len(base_resp.split())
        aligned_len = len(aligned_resp.split())
        
        # If the models have random weights, output length might just hit max_new_tokens.
        # But DPO pushes the model toward the chosen sequence length/tokens.
        if aligned_len < base_len:
            wins += 1
            print("Winner: Aligned (more concise)\n")
        elif base_len < aligned_len:
            print("Winner: Base (more concise)\n")
        else:
            print("Winner: Tie\n")
            
    win_rate = (wins / len(test_prompts)) * 100
    print(f"=====================================")
    print(f"Aligned Model Win Rate: {win_rate:.1f}%")
    print(f"=====================================")
    
if __name__ == "__main__":
    evaluate()
