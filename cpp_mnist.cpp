#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <cmath>


// some handy type aliases
using FloatType = float;
using FloatArray = std::vector<FloatType>;


// an MNIST sample holding the image (28*28 float values) and a label (0 or 1)
struct Sample {
	FloatArray image;
	FloatType target{};
};
using Dataset = std::vector<Sample>;


// load the MNIST dataset from a text file
Dataset load_dataset(const std::string& filepath) {
	Dataset dataset;

	std::ifstream file(filepath);
	if (!file) {
		std::cout << "Could not read dataset!" << std::endl;
		return dataset;
	}
	std::string tmp;
	char c;
	bool is_label = true;
	while (file.get(c)) {
		tmp += c;
		if (c == ',' || c == ';') {
			float val = std::stof(tmp);
			tmp.clear();
			if (is_label) {
				dataset.push_back(Sample());
				dataset.back().target = val;
				is_label = false;
			}
			else {
				dataset.back().image.push_back(val / 255);
			}
			if (c == ';') {
				is_label = true;
			}
		}
	}

	return dataset;
}


// show an MNIST image on the console
void imshow(const FloatArray& image) {
	for (size_t i = 0; i < image.size(); ++i) {
		std::cout << (image[i] > 0.5 ? "X" : " ");
		if (i > 0 && i % 28 == 0) {
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;
}


// create a weight vector containing random values
FloatArray random_weight(size_t num_elements) {
	FloatArray weight;
	weight.reserve(num_elements);
	for (size_t i = 0; i < num_elements; ++i) {
		weight.push_back(static_cast<FloatType>((0.01 * std::rand()) / RAND_MAX));
	}
	return weight;
}


// Unit is the main buildig block of the model, aka "neuron"
struct Unit {
	Unit(size_t a_num_input, bool a_has_activation)
		:has_activation(a_has_activation),
		weight(random_weight(a_num_input)) {
	}

	explicit Unit(FloatType a_value)
		: value(a_value) {
	}

	bool has_activation{};

	FloatArray weight;
	FloatType bias{};

	FloatType value{};

	FloatType delta{};
	FloatArray grad_weight;
	FloatType grad_bias{};
};


// RegressionModel contains multiple layers with Units, and outputs a single float value
struct RegressionModel {
	using Layer = std::vector<Unit>;

	// create a model with given number of units in input layer, number of units in hidden layers, and number of layers
	RegressionModel(size_t num_input, size_t num_hidden, size_t num_layer) {
		// input layer to store the inputs for each pass
		layer.push_back(Layer());
		for (size_t i = 0; i < num_input; ++i) {
			layer.back().push_back(Unit(0));
		}

		// hidden layers
		if (num_layer >= 2) {
			for (size_t i = 0; i < num_layer - 1; ++i) {
				layer.push_back(Layer());
				for (size_t j = 0; j < num_hidden; ++j) {
					layer.back().push_back(Unit(i == 0 ? num_input : num_hidden, true));
				}
			}
		}

		// last layer contains one single unit
		layer.push_back(Layer());
		layer.back().push_back(Unit(num_hidden, false));
	}

	// compute forward pass
	FloatType forward(const FloatArray& input_data) {
		// store inputs in first layer, as they are required in the backward pass
		for (size_t i = 0; i < input_data.size(); ++i) {
			layer[0][i].value = input_data[i];
		}

		// feed forward
		for (size_t i = 1; i < layer.size(); ++i) {
			Layer& curr_layer = layer[i];
			const Layer& input_layer = layer[i - 1];

			for (Unit& curr_unit : curr_layer) {
				FloatType pre_activation = curr_unit.bias;
				for (size_t j = 0; j < input_layer.size(); ++j) {
					pre_activation += input_layer[j].value * curr_unit.weight[j];
				}
				curr_unit.value = curr_unit.has_activation ? std::max<FloatType>(0, pre_activation) : pre_activation;
			}
		}

		// return output of single unit in last layer
		return layer.back()[0].value;
	}

	// compute backward pass
	void backward(FloatType error_signal) {
		// set the error signal (derivative of loss wrt last units pre-activation) for single unit in last layer
		layer.back()[0].delta = error_signal;

		// reversed index r counts from back, i is to be used to index the vector
		for (size_t r = 0; r < layer.size() - 1; ++r) {
			const size_t i = layer.size() - r - 1; // index to be used for the vector

			// going over all units of current layer
			Layer& curr_layer = layer[i];
			for (size_t j = 0; j < curr_layer.size(); ++j) {
				Unit& curr_unit = layer[i][j];
				// for all but last layer compute delta (incoming error signal)
				if (r > 0) {
					curr_unit.delta = 0;
					const Layer& next_layer = layer[i + 1]; // closer to the output
					for (size_t k = 0; k < next_layer.size(); ++k) {
						const Unit& next_unit = next_layer[k];
						curr_unit.delta += next_unit.delta * next_unit.weight[i];
					}
				}

				// compute gradient for weights and for bias (derivative of loss wrt each parameter)
				FloatArray grad;
				const Layer& prev_layer = layer[i - 1];
				for (size_t k = 0; k < prev_layer.size(); ++k) {
					const Unit& prev_unit = prev_layer[k];
					grad.push_back(curr_unit.delta * prev_unit.value);
				}
				curr_unit.grad_weight = grad;
				curr_unit.grad_bias = curr_unit.delta;
			}
		}
	}

	// do a small step in direction of the negative gradient, as this reduces the loss
	void step(FloatType lr) {
		for (size_t i = 1; i < layer.size(); ++i) {
			Layer& curr_layer = layer[i];
			for (Unit& unit : curr_layer) {
				for (size_t j = 0; j < unit.weight.size(); ++j) {
					unit.weight[j] -= lr * unit.grad_weight[j];
				}
				unit.bias -= lr * unit.grad_bias;
			}
		}
	}

	std::vector<Layer> layer;
};

FloatType classify(FloatType prediction) {
	return prediction > 0.5 ? 1 : 0;
}


// train, test, and analyze model
int main()
{
	// build the model
	RegressionModel model = RegressionModel(28 * 28, 10, 3);

	// training loop
	const Dataset dataset_train = load_dataset("mnist_train.txt");
	std::cout << "Training model..." << std::endl;
	std::cout << "Dataset size: " << dataset_train.size() << std::endl;
	for (size_t epoch = 0; epoch < 10; ++epoch) {
		for (size_t i = 0; i < dataset_train.size(); ++i) {
			const Sample& sample = dataset_train[i];
			FloatType prediction = model.forward(sample.image);
			FloatType loss = static_cast<FloatType>(0.5 * std::pow(sample.target - prediction, 2));
			FloatType error_signal = prediction - sample.target;
			model.backward(error_signal);
			model.step(static_cast<FloatType>(0.001));

			// don't print too often, this is slow
			if (i % 1000 == 0) {
				std::cout << "Epoch: " << epoch << " Sample: " << i << " Loss: " << loss << std::endl;
			}
		}
	}

	// check model accuracy on testset
	const Dataset dataset_test = load_dataset("mnist_test.txt");
	std::cout << "Testing model..." << std::endl;
	std::cout << "Dataset size: " << dataset_test.size() << std::endl;
	size_t correct_ctr = 0;
	for (const Sample& sample : dataset_test) {
		FloatType prediction = model.forward(sample.image);
		correct_ctr += classify(prediction) == sample.target ? 1 : 0;
	}
	FloatType accuracy = static_cast<FloatType>(correct_ctr) / dataset_test.size();
	std::cout << "Accuracy: " << accuracy << std::endl;

	// show testset samples and predictions
	for (const Sample& sample : dataset_test) {
		FloatType prediction = model.forward(sample.image);
		imshow(sample.image);
		std::cout << "Predicted: " << classify(prediction) << " (" << prediction << ") Target: " << sample.target << std::endl;
		std::cout << "Press ENTER to see next sample..." << std::endl;
		(void)std::getchar();
	}

	return 0;
}
