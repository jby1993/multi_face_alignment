#ifndef RANDOM_TOOL_H
#define RANDOM_TOOL_H
#include "random_num_generator.h"

class random_tool
{
public:
    random_tool();
    void random_choose_n_different_id_from_0_to_size(int n, int size, std::vector<int> &ids);
    int random_choose_num_from_1_to_size(int size);
private:
    base_generator_type m_random_gen;
};

#endif // RANDOM_TOOL_H
