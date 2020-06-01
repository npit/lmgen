from huggingface_lm import HuggingFaceLanguageModel
import json

if __name__ == "__main__":
	epochs = 2
	batch_size = 8
	sequence_length = 8

	with open("") as f:
		data = json.load(f)


	hlm = HuggingFaceLanguageModel(batch_size=batch_size, sequence_length=sequence_length)
	hlm.prepare_data(data)
	hlm.train(epochs)
	hlm.save(".")