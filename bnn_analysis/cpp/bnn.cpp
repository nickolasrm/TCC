#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <limits>
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
 * @brief Create a bit vector
 *
 */
class BitArray : public Array<unsigned>
{
public:
    int packed_length;
    static const unsigned MAX = std::numeric_limits<unsigned>::max();
    static const unsigned BITS = std::numeric_limits<unsigned>::digits;

public:
    BitArray(int size) : Array(size)
    {
        this->packed_length = calc_packs(size);
    }

public:
    /**
     * @brief Calculate the number of packs needed to store a bit vector
     *
     * @param size required bits
     * @return int required packs
     */
    static int calc_packs(int size)
    {
        return ceil(size / (float)BitArray::BITS);
    }
    /**
     * @brief Get the pack index of a bit vector
     * 
     * @param index bit index
     * @return int pack index
     */
    static int pack_index(int index)
    {
        return floor(index / (float)BitArray::BITS);
    }
    /**
     * @brief Get the bit index of the target pack
     *
     * @param index bit index
     * @return int bit index of the target pack
     */
    static int pack_bit_index(int index)
    {
        return index % sizeof(unsigned);
    }

public:
    /**
     * @brief Get a bit from a bit vector
     *
     * @param index bit index
     * @return int bit value
     */
    int get(int index)
    {
        unsigned pack = this->data[this->pack_index(index)];
        index = this->pack_bit_index(index);
        return (pack >> index) & 1;
    }
    /**
     * @brief Set a bit in a bit vector
     *
     * @param index bit index
     * @param value bit value
     */
    void set(int index, int value)
    {
        int pack_index = this->pack_index(index);
        unsigned pack = this->data[pack_index];
        index = this->pack_bit_index(index);
        pack &= ~(1 << index);
        pack |= value << index;
        this->data[pack_index] = pack;
    }

public:
    static BitArray *from_primitive(unsigned *array, int length)
    {
        BitArray *bit_array = new BitArray(length);
        for (int i = 0; i < length; i++)
            bit_array->set(i, array[i] < 0);
        return bit_array;
    }
};

/**
 * @brief Bitwise layer
 *
 */
class BitLayer
{
public:
    int *b;
    BitArray *w;
    BitArray *output;
    int units;
    int inputs;

public:
    BitLayer(int units, int inputs)
    {
        this->units = units;
        this->inputs = inputs;
        this->b = new int[units];
        int padded_inputs = BitArray::calc_packs(inputs) * BitArray::BITS;
        /* This step is required for removing the need of bit masks */
        this->w = new BitArray(units * padded_inputs);
        this->output = new BitArray(units);
    }
    ~BitLayer()
    {
        delete[] this->b;
        delete this->w;
        delete this->output;
    }

public:

    /**
     * @brief Perform a bitwise dot product using XOR and popcount
     *
     * @param input input vector
     */
    void dot_product(BitArray *input)
    {
        unsigned *w = this->w->data;
        unsigned *input_data = input->data;
        int *b = this->b;
        int n_packed_inputs = input->packed_length;
        int n_inputs = this->inputs;
        int units = this->units;
        BitArray *output = this->output;

        for (int i = 0; i < units; i++)
        {
            int minus_ones = 0;
            for (int j = 0; j < n_packed_inputs; j++)
                minus_ones += __builtin_popcount(input_data[j] ^ w[j]);
            int plus_ones = n_inputs - minus_ones;
            output->set(i, signbit(plus_ones - minus_ones + b[i]));

            w += input->packed_length;
        }
    }
};

/**
 * @brief Bitwise neural network
 * 
 */
class NeuralNetwork
{
public:
    Array<BitLayer *> *layers;
    int inputs;

public:
    NeuralNetwork(int inputs, Array<int> *units)
    {
        this->layers = new Array<BitLayer *>(units->length);
        for (int i = 0; i < units->length; i++)
        {
            this->layers->data[i] = new BitLayer(units->data[i], inputs);
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
     * @brief Bitwise feed forward
     *
     * @param input Network input
     * @return Network output
     */
    BitArray *feed_forward(BitArray *input)
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
    // Create a BNN
    NeuralNetwork *bnn = new NeuralNetwork(inputs, Array<int>::from_primitive(units, layers));
    // Feed forward
    BitArray *input_arr = new BitArray(inputs);

    // Measure time
    clock_t t;
    t = clock();
    BitArray *output = bnn->feed_forward(input_arr);
    t = clock() - t;
    double time_taken = ((double)t) / CLOCKS_PER_SEC; // in seconds
    printf("%lf", time_taken);
}
