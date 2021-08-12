#!/bin/bash

cat > $1.job << END
#!/bin/bash
cd $PWD
source env/bin/activate
python scripts/incremental_learning/experiment_driver.py with threads=$2 &> $1.log
if [ $? -eq 0 ]; then rm $1.job; fi
END

chmod 0775 $1.job
tsp -N $2 $PWD/$1.job 
