/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <iostream>
#include <vector>

template<typename T>
struct STemplated {
    void printFirst() { std::cout << s_First << std::endl; }

    T s_First;
    std::vector<T> s_Second;
};

struct SSimple {
    void printFirst() { std::cout << s_First << std::endl; }

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
