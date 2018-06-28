/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <iostream>
#include <vector>

template<typename T>
struct STemplated {
    void printFirst(void) { std::cout << s_First << std::endl; }

    T s_First;
    std::vector<T> s_Second;
};

struct SSimple {
    void printFirst(void) { std::cout << s_First << std::endl; }

    int s_First;
    std::vector<int> s_Second;
};

int main(int, char**) {
    {
        SSimple obj;
        obj.printFirst();
    }

    {
        STemplated<int> obj;
        obj.printFirst();
    }

    return 0;
}
