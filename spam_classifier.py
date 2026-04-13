import torch
from transformers import GPT2ForSequenceClassification
import tiktoken

class SpamClassifier:
    """Spam Classifier exactly following Book Chapter 6 (LoRA fine-tuned GPT-2)"""
    
    def __init__(self, model_path="spam_classifier_model.pth"):
        print("Loading GPT-2 For Sequence Classification + your trained weights (Chapter 6 style)...")
        
        # Book Chapter 6 style: GPT2 for binary classification (spam/ham)
        self.model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)
        
        # Directly load your saved state_dict (the simplest and most reliable way)
        state_dict = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=False)   # strict=False to ignore LoRA extra keys
        self.model.eval()
        
        # Book Chapter 2/5/6: real tiktoken tokenizer
        self.tokenizer = tiktoken.get_encoding("gpt2")
        
        print("✅ Spam classifier model loaded successfully from spam_classifier_model.pth!")

    def predict(self, text: str):
        """Spam / Ham prediction + probability + math explanation"""
        # Tokenize exactly like book Chapter 6
        tokens = self.tokenizer.encode(text)
        input_ids = torch.tensor([tokens], dtype=torch.long)
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
            
        probs = torch.softmax(logits, dim=-1)
        spam_prob = probs[0][1].item()   # index 1 = spam
        
        label = "SPAM" if spam_prob > 0.5 else "HAM"
        
        explanation = f"""Classification confidence: {spam_prob:.1%}"""

        return label, spam_prob, explanation