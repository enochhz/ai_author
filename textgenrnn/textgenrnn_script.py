from textgenrnn import textgenrnn
textgen = textgenrnn(weights_path='shakespeare_weights.hdf5', vocab_path='shakespeare_vocab.json', config_path='shakespeare_config.json')

textgen.generate_samples(n=1, max_gen_length=300)
textgen.generate_to_file('shakespeare_texts.txt', n=1, max_gen_length=300)
