# Colorize-then-fit baseline: a single Plenoxels run whose training target is the
# teacher's colorized images. There is no distillation and no second stage -- the
# radiance field never sees the monochrome inputs. Use it to compare against the
# two-stage pipeline in try_llff.sh, which starts from the monochrome inputs and
# distils colour into a fixed geometry.
SCENE=$1
DATA_ROOT=${2:-../data/llff}

data_type=llff
ckpt_baseline=ckpt_baseline/${data_type}/${SCENE}
data_dir=${DATA_ROOT}/${SCENE}

python opt.py -t ${ckpt_baseline} ${data_dir} \
                -c configs/llff_baseline.json

python render_imgs.py ${ckpt_baseline}/ckpt.npz ${data_dir} \
                --render_path
