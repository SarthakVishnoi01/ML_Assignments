#!/bin/bash

# TO RUN:
# 1. Put your files in submissions directory and run the script as follows:
# ./evaluate_q1.sh <path_to_data_dir> <path_to_sandbox_dir> <entry_number> <path_to_submissions_dir>
# Example:
# ./evaluate_q1.sh $HOME/data $HOME/sandbox 2016ANZ8048 $HOME/submissions

run()
{
	: '
        Args:
	    	$1 file name
	    	$2 part
	    	$3 train data
	    	$4 test data
	    	$5 output file
    '
	chmod +x $1
    ./$1 $2 $3 $4 $5
}

compute_score()
{
    : '
        Compute score as per predicted values and write to given file
        $1 python_file
        $2 targets
        $3 predicted
        $4 outfile
    '
    python3 $1 $2 $3 $4
}

main()
{
    : '
        $1: data_dir
        $2: sandbox_dir
        $3: entry_number
        $4: submissions_dir
    '
    main_dir=`pwd`
    unzip $4/$3.zip -d $2
    # Run Q1
    cd $2/$3
    for part in a b c; do
        run linreg $part $1/msd_train.csv $1/msd_test.csv $2/$3/results_$part
        compute_score $main_dir/compute_error.py $1/msd_test.csv $2/$3/results_$part
    done 
}

main $1 $2 $3 $4
