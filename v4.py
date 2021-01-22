#!/usr/bin/env python
# -*- coding: utf-8 -*-
from v4_funcs import *





def work_flow():
    print('crop > txt > train > tile > test > final')

def main():
    # step 1 cut flaw crops
    ask_stale_imgs(CROPS_DIR)
    get_crop_objs_from_train(TRAIN_DIR, JSON_PATH) # write to disk each 100 imgs, to prevent men boom
    exit()

    # step 2 make txt labels for crops
    make_txt_from_crops(CROPS_DIR, TXT_DIR)
    exit()

    # step 3 train
    print('train')
    # local: python train.py --img 640 --batch 2 --epochs 2 --data v4.yaml  --worker 1  --weights kaggle_1000.pt
    # sky:   python train.py --img 640 --batch 128 --epochs 100  --data v4.yaml --weights kaggle_1000.pt
    exit()

    # step 4 cut test big img to tiles
    ask_stale_imgs(TILES_DIR)
    crop_holes = {
        0: get_tiles([8192, 6000], TILE_SIZE, LAP_SIZE),
        1: get_tiles([4096, 3500], TILE_SIZE, LAP_SIZE)
    }
    get_tile_objs_from_test(TEST_DIR, crop_holes)
    exit()

    # step 5 test
    print('test')
    # local: python test.py --img 640 --data v4.yaml --save-json  --weights kaggle_1000.pt  --conf 0.5 --task test  --batch 2
    # sky:   python test.py --img 640 --data v4.yaml --save-json  --weights kaggle_1000.pt  --conf 0.5 --task test  --batch 128
    exit()

    # step 6 get result
    tile_json_path = r'yolov5-master\runs\test\exp11\kaggle_1000_predictions.json'
    final_json_path = 'ds/v4/final.json'
    get_final_json(tile_json_path, final_json_path)
    exit()


if __name__ == "__main__":
    main()
