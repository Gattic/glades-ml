Public CPU package/export and read-only deployment regression (after building the library):
`bash unit-tests/test.sh package-consumer`. See
[`tests/package-consumer/README.md`](../tests/package-consumer/README.md) for the
explicit-directory loading contract and isolated outputs. This selector uses
CMake and the C++ consumer only; it has no Python dependency.

Step schedule + clip:
sh test.sh nn-bench --dataset datasets/rnn.csv --epochs 200 --hidden 8 --repeats 3 --lr 0.05 --lr-schedule step --step-size 50 --gamma 0.5 --clip-norm 5

Cosine schedule:
sh test.sh nn-bench --lr-schedule cosine --tmax 200 --min-mult 0.1
