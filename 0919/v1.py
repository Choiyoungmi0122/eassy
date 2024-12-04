import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from datasets import load_from_disk
from torch.utils.data import DataLoader

torch.cuda.empty_cache()

# Custom collate function
def custom_collate_fn(batch, tokenizer):
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    attention_mask = [torch.tensor(item['attention_mask']) for item in batch]
    
    # Ensure all sequences have the same length
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {'input_ids': input_ids, 'attention_mask': attention_mask}


# PyTorch Lightning module for training
class GemmaLightningTrainer(pl.LightningModule):
    def __init__(self, model, train_dataset, valid_dataset, training_args, tokenizer):
        super().__init__()
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.training_args = training_args
        self.tokenizer = tokenizer

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch, labels=batch["input_ids"])
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch, labels=batch["input_ids"])
        loss = outputs.loss
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=3e-5)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.training_args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=lambda x: custom_collate_fn(x, self.tokenizer)
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.training_args.per_device_eval_batch_size,
            collate_fn=lambda x: custom_collate_fn(x, self.tokenizer)
        )


# Main function for VSCode execution
def main():
    # Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    # Load the preprocessed datasets (train, valid)
    train_dataset = load_from_disk('/Users/choiyoungmi/Desktop/code/essay/0910/tokenized_data/train')
    valid_dataset = load_from_disk('/Users/choiyoungmi/Desktop/code/essay/0910/tokenized_data/valid')

    # Load the tokenizer and model
    model_name = "google/gemma-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token="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")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Auto-distribute across GPUs
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        token="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"
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./output_gemma_lightning_3080",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,  
        num_train_epochs=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs_3080',
        save_steps=2000,  
        logging_steps=500,  
        fp16=True,  
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none"
    )

    # Initialize the Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=training_args.num_train_epochs,
        accelerator="gpu",
        devices=1,
        strategy=DDPStrategy(find_unused_parameters=False),
        precision=32,  
        log_every_n_steps=100,
        default_root_dir="./output_gemma_lightning_3080"
    )

    # Initialize the custom trainer
    gemma_trainer = GemmaLightningTrainer(model, train_dataset, valid_dataset, training_args, tokenizer)

    # Start training
    trainer.fit(gemma_trainer)

    # Example inference after training
    input_text = "저는 항상 기술 혁신과 발전에 큰 관심을 가지고 있었습니다."
    similar_sentences = generate_similar_sentences(input_text, model, tokenizer)
    for idx, sentence in enumerate(similar_sentences, 1):
        print(f"Generated Sentence {idx}: {sentence}")

    # Save model and tokenizer after fine-tuning
    model.save_pretrained("./output_gemma_lightning_3080")
    tokenizer.save_pretrained("./output_gemma_lightning_3080")


# Function to generate similar sentences
def generate_similar_sentences(input_text, model, tokenizer, top_k=3):
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(**input_ids, max_new_tokens=50, num_return_sequences=top_k)
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


if __name__ == "__main__":
    main()
