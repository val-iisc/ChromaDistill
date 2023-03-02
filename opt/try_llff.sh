SCENE=fern
STYLE=1000 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_1.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_1.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1



SCENE=flower
STYLE = 1000 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_1.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_1.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1



SCENE=fortress
STYLE = 1000 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_1.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_1.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1


SCENE=giants
STYLE = 1000 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_1.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_1.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1



SCENE=horns
STYLE = 1000 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_1.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_1.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1



SCENE=leaves
STYLE = 1000 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_1.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_1.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1



SCENE=orchids
STYLE = 1000 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_1.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_1.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1



SCENE=pasta
STYLE = 1000 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_1.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_1.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1



SCENE=room
STYLE = 1000 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_1.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_1.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1



SCENE=trex
STYLE = 1000 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_1.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_1.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1


SCENE=cake
STYLE = 1000 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_1.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_1.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1


SCENE=chilli_paste
STYLE = 1000 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_1.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_1.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1


SCENE=different_room
STYLE = 1000 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_1.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_1.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1



SCENE=fish
STYLE = 1000 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_1.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_1.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1


SCENE=glass
STYLE = 1000 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_1.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_1.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1


SCENE=guitars
STYLE = 1000 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_1.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_1.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1



SCENE=shiny_budda
STYLE = 1000 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_1.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_1.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1


SCENE=shrimp
STYLE = 1000 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_1.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_1.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1


SCENE=small_leaves
STYLE = 1000 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_1.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_1.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1


SCENE=three_budda
STYLE = 1000 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_1.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_1.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 1



SCENE=fern
STYLE=1001 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_2.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_2.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2



SCENE=flower
STYLE=1001 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_2.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_2.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2



SCENE=fortress
STYLE = 1001 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_2.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_2.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2


SCENE=giants
STYLE = 1001 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_2.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_2.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2



SCENE=horns
STYLE = 1001 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_2.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_2.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2



SCENE=leaves
STYLE = 1001 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_2.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_2.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2



SCENE=orchids
STYLE = 1001 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_2.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_2.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2



SCENE=pasta
STYLE = 1001 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_2.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_2.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2



SCENE=room
STYLE = 1001 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_2.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_2.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2



SCENE=trex
STYLE = 1001 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_2.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_2.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2


SCENE=cake
STYLE = 1001 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_2.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_2.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2


SCENE=chilli_paste
STYLE = 1001 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_2.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_2.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2


SCENE=different_room
STYLE = 1001 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_2.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_2.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2



SCENE=fish
STYLE = 1001 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_2.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_2.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2


SCENE=glass
STYLE = 1001 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_2.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_2.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2


SCENE=guitars
STYLE = 1001 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_2.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_2.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2



SCENE=shiny_budda
STYLE = 1001 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_2.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_2.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2


SCENE=shrimp
STYLE = 1001 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_2.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_2.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2


SCENE=small_leaves
STYLE = 1001 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_2.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_2.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2


SCENE=three_budda
STYLE = 1001 
data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_arf=ckpt_arf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff_teacher_2.json
fi

python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2

python opt_style.py -t ${ckpt_arf} ${data_dir} \
                -c configs/llff_fixgeom_teacher_2.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 

python render_imgs.py ${ckpt_arf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave --teacher 2

