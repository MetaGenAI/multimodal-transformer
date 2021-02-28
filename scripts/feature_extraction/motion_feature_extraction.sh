
folder=$1
py=python
n=$(nproc)

mpirun -n $n $py ./scripts/feature_extraction/process_motions.py $1
mpirun -n 1 $py ./scripts/feature_extraction/extract_transform.py $1 --feature_name pkl_joint_angles_mats --transforms scaler
mpirun -n $n $py ./scripts/feature_extraction/apply_transform.py $1 --feature_name pkl_joint_angles_mats --transform_name scaler --new_feature_name joint_angles_scaled
