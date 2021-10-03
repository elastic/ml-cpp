#!/bin/bash

CYCLE=1
# Do half the test cycles in one order
while [ $CYCLE -le 5 ]
do
    TEST1NS=`./vfprog 1 | tail -1 | awk '{ print $3 }'`
    TEST2NS=`./vfprog 2 | tail -1 | awk '{ print $3 }'`
    TEST3NS=`./vfprog 3 | tail -1 | awk '{ print $3 }'`
    TEST4NS=`./vfprog 4 | tail -1 | awk '{ print $3 }'`
    TEST5NS=`./vfprog 5 | tail -1 | awk '{ print $3 }'`
    TEST6NS=`./vfprog 6 | tail -1 | awk '{ print $3 }'`
    TEST7NS=`./vfprog 7 | tail -1 | awk '{ print $3 }'`
    TEST8NS=`./vfprog 8 | tail -1 | awk '{ print $3 }'`
    TEST9NS=`./vfprog 9 | tail -1 | awk '{ print $3 }'`
    echo `uname -n`",$TEST1NS,$TEST2NS,$TEST3NS,$TEST4NS,$TEST5NS,$TEST6NS,$TEST7NS,$TEST8NS,$TEST9NS"
    CYCLE=`expr $CYCLE + 1`
done
# Do the other half of the test cycles in reverse order
# in case the ordering creates any bias
while [ $CYCLE -le 10 ]
do
    TEST9NS=`./vfprog 9 | tail -1 | awk '{ print $3 }'`
    TEST8NS=`./vfprog 8 | tail -1 | awk '{ print $3 }'`
    TEST7NS=`./vfprog 7 | tail -1 | awk '{ print $3 }'`
    TEST6NS=`./vfprog 6 | tail -1 | awk '{ print $3 }'`
    TEST5NS=`./vfprog 5 | tail -1 | awk '{ print $3 }'`
    TEST4NS=`./vfprog 4 | tail -1 | awk '{ print $3 }'`
    TEST3NS=`./vfprog 3 | tail -1 | awk '{ print $3 }'`
    TEST2NS=`./vfprog 2 | tail -1 | awk '{ print $3 }'`
    TEST1NS=`./vfprog 1 | tail -1 | awk '{ print $3 }'`
    echo `uname -n`",$TEST1NS,$TEST2NS,$TEST3NS,$TEST4NS,$TEST5NS,$TEST6NS,$TEST7NS,$TEST8NS,$TEST9NS"
    CYCLE=`expr $CYCLE + 1`
done

