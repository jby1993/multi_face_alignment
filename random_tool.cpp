#include "random_tool.h"
#include <time.h>
#include <set>
random_tool::random_tool()
{
    m_random_gen.seed(time(0));
}

void random_tool::random_choose_n_different_id_from_0_to_size(int n, int size, std::vector<int> &ids)
{
    std::vector<bool> is_choosed(size+1,false);
    ids.clear();
    for(int i=0;i<n;i++)
    {
        int id=rand()%(size+1-i);
        int num=0;
        for(int j=0;j<is_choosed.size();j++)
        {
            if(!is_choosed[j])
            {
                num++;
                if(num==id+1)
                {
                    ids.push_back(j);
                    is_choosed[j]=true;
                    break;
                }
            }
        }
    }


    uniform_int_distribution uniform_int_random(m_random_gen,uniform_int_distribution_type(0,size));
    std::set<int> different_ids;
    for(int i=0;i<n;i++)
    {
        int id = uniform_int_random();
        if(different_ids.count(id)!=0)
        {
            i--;
            continue;
        }
        else
            different_ids.insert(id);
    }
    ids.clear();
    for(std::set<int>::iterator iter=different_ids.begin();iter!=different_ids.end();iter++)
    {
        ids.push_back(*iter);
    }
}

int random_tool::random_choose_num_from_1_to_size(int size)
{
    uniform_int_distribution uniform_int_random(m_random_gen,uniform_int_distribution_type(1,size));
    return uniform_int_random();
}
