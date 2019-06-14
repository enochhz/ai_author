from textgenrnn import textgenrnn
import sys
textgen = textgenrnn(weights_path=sys.argv[1], vocab_path=sys.argv[2], config_path=sys.argv[3])

textgen.generate_samples(n=1, max_gen_length=300)
textgen.generate_to_file('shakespeare_texts.txt', n=1, max_gen_length=300)
