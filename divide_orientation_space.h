#ifndef DIVIDE_ORIENTATION_SPACE_H
#define DIVIDE_ORIENTATION_SPACE_H


class divide_orientation_space
{
public:
    divide_orientation_space();
    int get_divide_num(){return m_divide_num;}
    //using acr angle of x,y,z axis to compute belong to which subspace(left down origine)
    int compute_orientation_id(float ax, float ay, float az);
private:
    int m_divide_num;

};

#endif // DIVIDE_ORIENTATION_SPACE_H
