SCENE=$1
DATA_ROOT=${2:-../data/llff}

data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_distill=ckpt_distill/${data_type}/${SCENE}
data_dir=${DATA_ROOT}/${SCENE}

# Stage 1: fit the radiance field to the monochrome inputs
if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path

# Stage 2: distil colour from the teacher images, keeping the geometry fixed
python opt_style.py -t ${ckpt_distill} ${data_dir} \
                -c configs/llff_fixgeom_teacher.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3

python render_imgs.py ${ckpt_distill}/ckpt.npz ${data_dir} \
                    --render_path
