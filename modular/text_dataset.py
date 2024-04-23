import torch
import transformers
from torch.utils.data import Dataset
from collections import Counter
from jaxtyping import Float

class TextDataset(Dataset):
	def __init__(
			self, 
			text: str, 
			tokenizer: transformers.PreTrainedTokenizer,
			n_context: int,
			ensure_n_context_match: bool = True,
		):
		# add 1 to n_context to account for the target token
		n_context += 1

		# tokenize the text
		tokenized_text: list[int] = tokenizer.encode(text)
		self.total_tokens: int = len(tokenized_text)

		# trim the last tokens to make sure the length is a multiple of n_context
		if ensure_n_context_match:
			tokenized_text = tokenized_text[:-(len(tokenized_text) % n_context)]
			self.total_tokens = len(tokenized_text)

		# split the text into examples of length n_context
		# this means that text will often start in the middle of a sentence
		# in reality, we might want to do this a bit smarter
		self.examples: list[list[int]] = [
			tokenized_text[i:i+n_context] 
			for i in range(0, len(tokenized_text), n_context)
		]

	def __len__(self) -> int:
		return len(self.examples)
	
	def __getitem__(self, i: int) -> Float[torch.Tensor, "n_ctx"]:
		return torch.tensor(self.examples[i], dtype=torch.long, device = 'cpu')
	
	def example_lengths(self) -> Counter[int]:
		return Counter(len(ex) for ex in self.examples)
	