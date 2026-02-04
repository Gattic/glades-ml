Step schedule + clip:
sh test.sh nn-bench --dataset datasets/rnn.csv --epochs 200 --hidden 8 --repeats 3 --lr 0.05 --lr-schedule step --step-size 50 --gamma 0.5 --clip-norm 5

Cosine schedule:
sh test.sh nn-bench --lr-schedule cosine --tmax 200 --min-mult 0.1
