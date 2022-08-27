#include <iostream>
#include <random>

int main(){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    std::cout<<dis(gen)<<std::endl;
    return 0;
}