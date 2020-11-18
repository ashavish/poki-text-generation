from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf

SEED = 20
tf.random.set_seed(SEED)


data_path = "./poki.txt"
text = open(data_path, 'rb').read().decode(encoding='utf-8')
input_sequence = " ".join(text.split()[:50])

input_sequence = "I Love The Zoo : roses are red, violets are blue. i love the zoo. do you? The scary forest. : the forest is really haunted. i believe it to be so. but then we are going camping. A Hike At School :"


MAX_LEN = 200

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
GPT2 = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

#view model parameters
GPT2.summary()

# Greedy search

input_ids = tokenizer.encode(input_sequence, return_tensors='tf')

greedy_output = GPT2.generate(input_ids, max_length = MAX_LEN)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens = True))

# Beam Search outputs
beam_outputs = GPT2.generate(
    input_ids, 
    max_length = MAX_LEN, 
    num_beams = 5, 
    no_repeat_ngram_size = 2, 
    num_return_sequences = 5, 
    early_stopping = True
)

print('')
print("Output:\n" + 100 * '-')

for i, beam_output in enumerate(beam_outputs):
      print("{}: {}".format(i, tokenizer.decode(beam_output, skip_special_tokens=True)))


# Top K sampling
sample_output = GPT2.generate(
                             input_ids, 
                             do_sample = True, 
                             max_length = MAX_LEN, 
                             top_k = 10, 
                             temperature = 0.7
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens = True))      
