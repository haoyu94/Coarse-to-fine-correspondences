for N_POINTS in 250 500 1000 2500 5000
do
python scripts/benchmark_registration.py --source_path ./snapshot/tdmatch_enc_dec_test/3DMatch --benchmark 3DMatch --n_points $N_POINTS
done

