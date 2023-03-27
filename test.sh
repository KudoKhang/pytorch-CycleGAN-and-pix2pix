#!bin/bash
for epoch_number in {690..760..10}
do
    python3 test.py --dataroot ./datasets/ACNE04_PIX2PIX_FINAL_512_AUG \
                    --name removal_pix2pix_512_aug \
                    --model pix2pix --direction AtoB \
                    --crop_size 512 \
                    --load_size 512 \
                    --epoch ${epoch_number}
done