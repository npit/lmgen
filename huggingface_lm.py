import transformers
from transformers.optimization import AdamW
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from torch.utils.data import TensorDataset
import torch
from transformers import get_linear_schedule_with_warmup

from lm import LanguageModel, Dataset
import logging

import time
import datetime
from tqdm import tqdm

from os.path import exists
import pickle
import math

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
        print("Making tokenizer")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, output_hidden_states=True)
        # seqlen cannot be larger than 512 tokens
        if sequence_length > 512:
            raise ValueError("Sequence length: {sequence_length} is larger than the allowed max of 512")
        self.sequence_length = sequence_length
        self.batch_size = batch_size

        print("Making model")
        # self.model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")


    def tokenize_sentences(self, sentences):
        """
        Perform tokenization with a huggingface tokenizer

        sentences: List of strings
        Returns:
            tokenized_sentences: List of lists of tokens
        """
        print("Tokenizing sentences")
        filename = "data-tokenized.pkl"
        tok_sentences = []
        if exists(filename):
            print("Loading")
            # tok_sentences = torch.load(filename)
            with open(filename, "rb") as f:
                tok_sentences = pickle.load(f)
        else:
            tok_sentences = []
            # tokenize
            for sent in tqdm(sentences):
                tok_sentences.append(self.tokenizer.tokenize(sent))
            with open(filename, "wb") as f:
                pickle.dump(tok_sentences, f)
            # torch.save(filename)
        return tok_sentences

    def encode_sentences(self, data):

        print("Encoding sentences")
        filename = "data-encoded.pkl"
        if exists(filename):
            print("Loading")
            with open(filename, "rb") as f:
                input_ids, attention_masks = pickle.load(f)
        else:
            # tokenize sentences
            tokenized_sentences = self.tokenize_sentences(data)
            del data

            input_ids = []
            attention_masks = []
            for sent in tqdm(tokenized_sentences):
                # Tokenize sentence and add `[CLS]` and `[SEP]` tokens.
                encoding = self.tokenizer.encode_plus(sent, add_special_tokens=True, max_length=self.sequence_length,
                    return_attention_mask=True, return_tensors='pt', pad_to_max_length=True)
                input_ids.append(encoding["input_ids"])
                attention_masks.append(encoding["attention_mask"])
            del tokenized_sentences

            with open(filename, "wb") as f:
                pickle.dump((input_ids, attention_masks), f)

        # to torch tensors
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        return (input_ids, attention_masks)

    def prepare_data(self, data, only_eval=False):
        """Input data preparation function"""

        print("Preparing data")
        # perform sentence splitting
        sentences = self.tokenize_sentences(data)
        max_sentence_length = max(len(sent) for sent in sentences)
        # apply maximum sentence length 
        if self.sequence_length is None:
            self.sequence_length = max_sentence_length
        elif self.sequence_length > max_sentence_length:
            self.sequence_length = max_sentence_length

        # max sequence length in BERT
        self.sequence_length = min(self.sequence_length, 512)
        print("Encoding sentences to tokens and attention masks")
        self.input_ids, self.attention_masks = self.encode_sentences(data)

        print("Making dataloader")
        if only_eval:
            self.dataloader = torch.utils.data.DataLoader(TensorDataset(self.input_ids, self.attention_masks), batch_size=self.batch_size, shuffle=False)
        else:
            self.dataloader = torch.utils.data.DataLoader(TensorDataset(self.input_ids, self.attention_masks), batch_size=self.batch_size, shuffle=True)

    def train(self, num_epochs):
        device = configure_device()


        optimizer = optimizer = AdamW(self.model.parameters(), lr = 2e-5, eps = 1e-8 )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        t0 = time.time()
        total_train_loss = 0

        for ep in range(num_epochs):
            for step, batch in enumerate(self.dataloader):
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
                # import ipdb; ipdb.set_trace()
                outputs = self.model(b_input_ids,masked_lm_labels=b_input_ids,
                                    token_type_ids=None, attention_mask=b_input_mask)
                loss, predictions = outputs[:2]

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
                scheduler.step(loss)
                if step % 5 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)
                    # Report progress.
                    print('Epoch {},  Batch {:>5,}  of  {:>5,}. Local - accum loss: {} - {}    Elapsed: {:}.'.format(ep, step, math.ceil(len(self.dataloader) / self.batch_size), loss.item(), total_train_loss, elapsed))
            torch.save(self.model, "model_ep_{}".format(ep))


        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(self.dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

    def save(self, ddir):
        torch.save(model.state_dict(), ddir + self.name)
    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def encode_eval_data(self):
        outputs = torch.zeros((len(self.input_ids), 768))
        global_idx = 0

        res = []
        device = configure_device()
        self.model.eval()
        with torch.no_grad():
            with tqdm.tqdm(total=len(self.dataloader)) as pbar:
                for i, batch in enumerate(self.dataloader):

                    b_input_ids = batch[0].to(device)
                    b_input_mask = batch[1].to(device)

                    out = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

                    seq_vectors = outputs[1]
                    batch_len = len(seq_vectors)

                    global_ending_idx = global_idx+batch_len
                    outputs[global_idx:(global_ending_idx), :] = seq_vectors
                    global_idx = global_ending_idx

                    if i % 1000 == 0 and i > 0:
                        torch.save(outputs, "results_{}".format(i))
                    pbar.update()
