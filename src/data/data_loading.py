import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Any, Dict, Tuple
from .tokenizer import CCSRSequenceTokenizer
from torch.nn.utils.rnn import pad_sequence

class CCSRDataset(Dataset):
    def __init__(self, input_ids, attention_mask, targets=None) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.targets = targets

    def __len__(self) -> int:
        if hasattr(self.input_ids, "size"):
            return self.input_ids.size(0)
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx]
        }
        if self.targets is not None:
            item["targets"] = self.targets[idx]
        return item

def dynamic_collate_fn(batch):
    # Convert each item's "input_ids" and "attention_mask" to tensors
    input_ids = [torch.tensor(item['input_ids'], dtype=torch.long) for item in batch]
    attention_masks = [torch.tensor(item['attention_mask'], dtype=torch.long) for item in batch]
    targets = torch.tensor([item['targets'] for item in batch], dtype=torch.long)
    
    # Pad sequences dynamically for the batch
    # Replace 0 with tokenizer.token2idx['[PAD]'] if available.
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    
    return {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_masks,
        'targets': targets
    }

def create_dataloaders(config: Any) -> Tuple[DataLoader, DataLoader, DataLoader, CCSRSequenceTokenizer]:
    data_path = config.data_path
    if not os.path.isabs(data_path) and not os.path.exists(data_path):
        alt_path = os.path.join(os.path.dirname(__file__), data_path)
        if os.path.exists(alt_path):
            data_path = alt_path
        else:
            raise FileNotFoundError(f"Data file not found: {config.data_path} (also tried {alt_path})")

    tokenizer = CCSRSequenceTokenizer(
        mode=config.tokenizer_mode,
        max_length=config.max_length
    )

    with open(data_path, "r") as f:
        raw_text = f.read()
    sequences = tokenizer.preprocess_data(raw_text)
    tokenizer.build_vocab(sequences)
    config.vocab_size = len(tokenizer.idx2token)

    inputs = tokenizer.prepare_transformer_inputs(
        sequences,
        windowing_strategy=config.windowing_strategy,
        window_size=config.window_size,
        overlap=config.overlap,
        target_mode=config.target_mode
    )

    dataset = CCSRDataset(inputs["input_ids"], inputs["attention_mask"], inputs["targets"])
    total_len = len(dataset)
    train_size = int(config.train_ratio * total_len)
    val_size = int(config.val_ratio * total_len)
    test_size = total_len - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=dynamic_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=dynamic_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=dynamic_collate_fn)
    return train_loader, val_loader, test_loader, tokenizer
