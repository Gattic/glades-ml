sh test.sh nn
sh test.sh nn-recurrent
sh test.sh nn-transformer
sh test.sh nn-bench --dataset datasets/rnn.csv --epochs 200 --hidden 8 --repeats 3 --lr 0.05 --lr-schedule step --step-size 50 --gamma 0.5 --clip-norm 5
sh test.sh cv
sh test.sh save-load

