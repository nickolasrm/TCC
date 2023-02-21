#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/**
 * @brief Create a array
 *
 * @tparam T Elements type
 */
template <class T>
class Array
{
public:
    T *data;
    int length;

public:
    Array(int length)
    {
        this->length = length;
        this->data = new T[length];
    }
    ~Array()
    {
        delete[] this->data;
    }

public:
    T get(int index)
    {
        return this->data[index];
    }
    void set(int index, T value)
    {
        this->data[index] = value;
    }

public:
    static Array *from_primitive(T *array, int length)
    {
        Array *new_array = new Array(length);
        for (int i = 0; i < length; i++)
            new_array->set(i, array[i]);
        return new_array;
    }
};

/**
 * @brief Regular dense layer
 *
 */
class Layer
{
public:
    float *b;
    Array<float> *w;
    Array<float> *output;
    int units;
    int inputs;
    int half_inputs;

public:
    Layer(int units, int inputs)
    {
        this->units = units;
        this->inputs = inputs;
        this->b = new float[units];
        this->w = new Array<float>(units * inputs);
        this->output = new Array<float>(units);
    }
    ~Layer()
    {
        delete[] this->b;
        delete this->w;
        delete this->output;
    }

public:
    /**
     * @brief Returns X if X > 0, else 0
     *
     * @param x
     * @return float
     */
    inline float relu(float x)
    {
        return x > 0 ? x : 0;
    }

    /**
     * @brief Perform a dot product
     *
     * @param input input vector
     */
    void dot_product(Array<float> *input)
    {
        // this method accesses the data directly for performance reasons
        float *w = this->w->data;
        float *input_data = input->data;
        float *b = this->b;
        float *output = this->output->data;
        int units = this->units;
        int n_inputs = this->inputs;

        for (int i = 0; i < units; i++)
        {
            float sum = 0;
            for (int j = 0; j < n_inputs; j++)
                sum += input_data[j] * w[j];
            output[i] = this->relu(sum + b[i]);

            w += n_inputs;
        }
    }
};

/**
 * @brief Float neural network
 *
 */
class NeuralNetwork
{
public:
    Array<Layer *> *layers;
    int inputs;

public:
    NeuralNetwork(int inputs, Array<int> *units)
    {
        this->layers = new Array<Layer *>(units->length);
        for (int i = 0; i < units->length; i++)
        {
            this->layers->data[i] = new Layer(units->data[i], inputs);
            inputs = units->data[i];
        }
        this->inputs = inputs;
    }
    ~NeuralNetwork()
    {
        delete this->layers;
    }

public:
    /**
     * @brief Dense feed forward
     *
     * @param input Network input
     * @return Network output
     */
    Array<float> *feed_forward(Array<float> *input)
    {
        for (int i = 0; i < this->layers->length; i++)
        {
            this->layers->get(i)->dot_product(input);
            input = this->layers->get(i)->output;
        }
        return input;
    }
};

int main(int argc, char **argv)
{
    // Parse argv arguments as an array of numbers (integers), except for the first
    // argument which is a different number
    assert(argc > 2 && "Usage: ./libfile <input> <units> <units> ...");
    int layers = argc - 2;
    int inputs = atoi(argv[1]);
    int *units = (int *)malloc((layers) * sizeof(int));
    for (int i = 2; i < argc; i++)
        units[i - 2] = atoi(argv[i]);
    // Create a NN
    NeuralNetwork *nn = new NeuralNetwork(inputs, Array<int>::from_primitive(units, layers));
    // Feed forward
    Array<float> *input_arr = new Array<float>(inputs);

    // Measure time
    clock_t t;
    t = clock();
    Array<float> *output = nn->feed_forward(input_arr);
    t = clock() - t;
    int time_taken = 1e6 * ((double)t) / CLOCKS_PER_SEC; // in seconds
    printf("%d", time_taken);
}
