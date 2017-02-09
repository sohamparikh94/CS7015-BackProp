import _pickle as pkl
import gzip, argparse
import numpy as np

''' Parse arguments from the command line. 
lr specifies the learning rate
momentum specifies the momentum in optimization algorithms that require momentum
num_hidden specifies the number of hidden layers (number of layers excluding the input and the output layer)
sizes is a list of sizes of the hidden layers
...to be continued
'''

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--lr", type=float)
	parser.add_argument("--momentum", type=float)
	parser.add_argument("--num_hidden", type=int)
	parser.add_argument("--sizes", nargs='+')
	parser.add_argument("--activation", choices=["tanh", "sigmoid"])
	parser.add_argument("--loss", choices=["sq", "ce"])
	parser.add_argument("--opt", choices=["gd","momentum","nag", "adam"])
	parser.add_argument("--batch_size")
	parser.add_argument("--anneal", choices = ["True", "False"])
	parser.add_argument("--save_dir")
	parser.add_argument("--expt_dir")
	parser.add_argument("--mnist")
	args = parser.parse_args()
	global learning_rate, momentum, num_hidden, sizes, activation, loss, opt_algo, batch_size, anneal, save_dir, expt_dir, mnist_location
	learning_rate = args.lr
	momentum = args.momentum
	num_hidden = args.num_hidden
	sizes = [int(x) for x in args.sizes[0].split(',')]
	if(len(sizes) != num_hidden):
		print("\nError: num_hidden should be equal to the number of integers specified in sizes")
		exit(0)
	activation = args.activation
	loss = args.loss
	opt_algo = args.opt
	batch_size = args.batch_size
	anneal = args.anneal
	save_dir = args.save_dir
	expt_dir = args.expt_dir
	mnist_location = args.mnist

def import_train_data():
	global train_set, valid_set, test_set, train_size, valid_size, test_size
	train_set, valid_set, test_set = pkl.load(gzip.open(mnist_location, 'rb'), encoding='latin1')
	train_size = len(train_set[0])
	valid_size = len(valid_set[0])
	test_set = len(test_set[0])


def get_batch():
	train_size = len(train_set[0])
	indices = np.random.randint(0,train_size, [batch_size])
	return train_set[0][indices], train_set[1][indices]

def init_weights():
	weights = [np.random.randn(784,sizes[0])]
	weights += [np.random.randn(sizes[i],sizes[i+1]) for i in range(0,num_hidden - 1)]
	weights += [np.random.randn(sizes[num_hidden - 1], 10)]
	bias = [np.random.randn(sizes[i]) for i in range(0,num_hidden)]
	bias += [np.random.randn(10)]

def sigmoid(x):
	return 1/(1+np.exp(-x))

def softmax(x):
	return np.exp(x)/np.sum(np.exp(x))

def forward_pass():
	batch_input, labels = get_batch()
	batch_output = np.zeros([batch_size, 10])
	for i in range(batch_size):
		batch_output[i][labels[i]] = 1
	current_layer_input = np.copy(batch_input)
	if activation == "tanh":
		for i in range(num_hidden + 1):
			current_layer_output = np.tanh(np.dot(current_layer_input, weights[i]) + bias[i])
			current_layer_input = np.copy(current_layer_output)
	else:
		for i in range(num_hidden + 1):
			current_layer_output = sigmoid(np.dot(current_layer_input, weights[i]) + bias[i])
			current_layer_input = np.copy(current_layer_output)
	label_probabilities = softmax(current_layer_input)

def main():
	np.random.seed(1234)
	parse_args()
	import_train_data()
	init_weights()
	# train_data()


if __name__ == "__main__":
	main()