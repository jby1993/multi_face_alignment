#include "divide_orientation_space.h"
#include <math.h>
divide_orientation_space::divide_orientation_space()
{
    m_divide_num = 3;
}

int divide_orientation_space::compute_orientation_id(float ax, float ay, float az)
{
    float angle=M_PI*40.0/180.0;
    //now using ay to judge
    if(ay<-angle)
        return 0;
    else if(ay<angle)
        return 1;
    else
        return 2;
}
