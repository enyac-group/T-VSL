python main.py --train_data_path ./data/vggsound \
        --mode test --test_data_path ./data/vggsound \
        --test_gt_path ./metadata/vggsound_duet_test.csv \
        --output_dir ./path/to/output/dir \
        --id vggsound_duet --model tvsl \
        --trainset vggsound_duet --num_class 221 \
        --testset vggsound_duet --epochs 100 \
        --batch_size 256 --init_lr 0.01 \
        --lr_schedule cos --multiprocessing_distributed \
        --ngpu 4 --port 11342 --ciou_thr 0.3 \
        --iou_thr 0.3 --save_visualizations \
        --load /path/to/pretrained/model/for/evaluation \
        --audioclip_ckpt_path ./pretrained_weights/AudioCLIP-Full-Training.pt