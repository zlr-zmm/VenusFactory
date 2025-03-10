export HF_ENDPOINT=https://hf-mirror.com
dataset=DeepLocBinary
pdb_type=AlphaFold2
plm_model=esm2_t6_8M_UR50D
lr=5e-4
python src/eval.py \
    --plm_model facebook/$plm_model \
    --dataset_config data/$dataset/"$dataset"_"$pdb_type"_HF.json \
    --learning_rate $lr \
    --num_epochs 100 \
    --batch_token 12000 \
    --patience 10 \
    --structure_seq foldseek_seq,ss8_seq \
    --gradient_accumulation_steps 8 \
    --output_root result \
    --output_dir debug/$dataset/$plm_model \
    --output_model_name "$pdb_type"_"$lr"_bt12k_ga8.pt