import transformers
from transformers.optimization import AdamW
from transformers import BertTokenizer, BertForMaskedLM
import torch

from lm import LanguageModel, Dataset
import logging

import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))



def configure_device():
	# If there's a GPU available...
	if torch.cuda.is_available():    

		# Tell PyTorch to use the GPU.    
		device = torch.device("cuda")

		print('There are %d GPU(s) available.' % torch.cuda.device_count())

		print('We will use the GPU:', torch.cuda.get_device_name(0))

	# If not...
	else:
		print('No GPU available, using the CPU instead.')
		device = torch.device("cpu")
	return device


class HuggingFaceLanguageModel(LanguageModel):

	def __init__(self, batch_size, sequence_length):
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
		# seqlen cannot be larger than 512 tokens
		if sequence_length > 512:
			raise ValueError("Sequence length: {sequence_length} is larger than the allowed max of 512")
		self.sequence_length = sequence_length
		self.batch_size = batch_size

		self.model = BertForMaskedLM.from_pretrained("bert-base-uncased")


	def prepare_data(self, data):
		"""Input data preparation function"""
		# perform sentence splitting
		sentences = self.tokenize_sentences(data)
		# apply maximum sentence length
		max_sentence_length = max(max(len(sent) for sent in sentences), 512)

		input_ids = []
		attention_masks = []
		# For every sentence...
		for sent in sentences:
			# Tokenize sentence and add `[CLS]` and `[SEP]` tokens.
			encoding = self.tokenizer.encode_plus(sent, add_special_tokens=True, max_length=max_sentence_length,
				return_attention_mask=True, return_tensors='pt')
			input_ids.append(encoding["input_ids"])
			attention_masks.append(encoding["attention_mask"])

		# to torch tensors
		input_ids = torch.cat(input_ids, dim=0)
		attention_masks = torch.cat(attention_masks, dim=0)
		self.dataloader = torch.utils.data.Dataloader(sentences, shuffle=True)

	def train(self, num_epochs):
		device = configure_device()


		optimizer = optimizer = AdamW(self.model.parameters(), lr = 2e-5, eps = 1e-8 )
		t0 = time.time()
		for ep in range(num_epochs):
			for step, batch in enumerate(self.dataloader):
				if step % 40 == 0 and not step == 0:
					# Calculate elapsed time in minutes.
					elapsed = format_time(time.time() - t0)
					
					# Report progress.
					print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

				b_input_ids = batch[0].to(device)
				b_input_mask = batch[1].to(device)

				# Always clear any previously calculated gradients before performing a
				# backward pass. PyTorch doesn't do this automatically because 
				# accumulating the gradients is "convenient while training RNNs". 
				# (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
				self.model.zero_grad()        

				# Perform a forward pass (evaluate the model on this training batch).
				# The documentation for this `model` function is here: 
				# https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
				# It returns different numbers of parameters depending on what arguments
				# arge given and what flags are set. For our useage here, it returns
				# the loss (because we provided labels) and the "logits"--the model
				# outputs prior to activation.
				loss = self.model(b_input_ids, 
									token_type_ids=None, 
									attention_mask=b_input_mask)

				# Accumulate the training loss over all of the batches so that we can
				# calculate the average loss at the end. `loss` is a Tensor containing a
				# single value; the `.item()` function just returns the Python value 
				# from the tensor.
				total_train_loss += loss.item()

				# Perform a backward pass to calculate the gradients.
				loss.backward()

				# Clip the norm of the gradients to 1.0.
				# This is to help prevent the "exploding gradients" problem.
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

				# Update parameters and take a step using the computed gradient.
				# The optimizer dictates the "update rule"--how the parameters are
				# modified based on their gradients, the learning rate, etc.
				optimizer.step()

				# Update the learning rate.
				scheduler.step()

		# Calculate the average loss over all of the batches.
		avg_train_loss = total_train_loss / len(train_dataloader)            
		
		# Measure how long this epoch took.
		training_time = format_time(time.time() - t0)

		print("")
		print("  Average training loss: {0:.2f}".format(avg_train_loss))
		print("  Training epcoh took: {:}".format(training_time))

	def save(self, ddir):
		torch.save(model.state_dict(), ddir + self.name)
	def load(self, path):
		self.model.load_state_dict(torch.load(path))