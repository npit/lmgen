from huggingface_lm import HuggingFaceLanguageModel
import sys
import json

if __name__ == "__main__":
    epochs = 10
    batch_size = 64
    sequence_length = 16

    try:
        data_file = sys.argv[1]
    except ValueError:
        print("Need json data file")
        exit(1)

    with open(data_file) as f:
        data = json.load(f)

    hlm = HuggingFaceLanguageModel(batch_size=batch_size, sequence_length=sequence_length)
    hlm.prepare_data(data)
    # hlm.train(epochs)
    # hlm.save(".")
    hlm.encode_eval_data()
