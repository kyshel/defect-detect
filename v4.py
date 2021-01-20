#!/usr/bin/env python
# -*- coding: utf-8 -*-
from v4_funcs import *


def work_flow():
    crops_train = get_crops_from_train()
    write_crops_to_disk(crops_train, CROPS_TRAIN_PATH)

    crops_val = get_crops_from_val()
    result = get_result(crops_val)

    make_submit(result)


def get_crop_objs_from_image(img, img_labels, crop_size=CROP_SIZE):
    img_width = img.shape[1]
    img_height = img.shape[0]
    crop_objs = []
    for index, row in img_labels.iterrows():
        label_index = row['category']
        bbox = row['bbox']

        cx = (bbox[2] + bbox[0]) / 2
        cy = (bbox[3] + bbox[1]) / 2
        base_x = int(cx - crop_size / 2)
        base_y = int(cy - crop_size / 2)

        crop_size_x, crop_size_y = crop_size, crop_size
        if base_x < 0: base_x = 0
        if base_y < 0: base_y = 0
        if base_x + crop_size > img_width: crop_size_x = img_width - base_x
        if base_y + crop_size > img_height: crop_size_y = img_height - base_y

        crop = img[base_y:base_y + crop_size_y, base_x:base_x + crop_size_x]
        crop_bbox = [bbox[0] - base_x + 1, bbox[1] - base_y + 1, bbox[2] - base_x + 1, bbox[3] - base_y + 1]
        # crop_bbox=[bbox[0]-base_x,bbox[1]-base_y,bbox[2]-base_x,bbox[3]-base_y]

        img_name_witout_ext = os.path.splitext(row['name'])[0]
        x, y, w, h = convert([crop_size_x, crop_size_y], crop_bbox)
        filename = '{0}__{1}__{2}__{3}__{4}__{5}__{6}__{7}__{8}__{9}.jpg'.format(
            img_name_witout_ext, base_x, base_y, crop_size_x, crop_size_y, label_index, x, y, w, h
        )  # alternate '__'.join(x,y,w,h)

        # draw some graph
        # x = int(crop_bbox[0])
        # y = int(crop_bbox[1])
        # w = int(crop_bbox[2] - crop_bbox[0])
        # h = int(crop_bbox[3] - crop_bbox[1])
        #
        # draw_cross(crop, (x, y))
        # color = flaw(label_index)['color']
        # cv2.rectangle(crop, (x, y, w, h), color, 1)
        # cv2.rectangle(crop, (0, 0,
        #                     crop_size_x, crop_size_y), color, 10)
        # text = "{},x{},y{},w{},h{}".format(flaw(label_index)['name'], x, y, w, h)
        # cv2.putText(crop, text, (x - 40, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # build flaw_crop
        crop_obj = {'filename': filename, 'data': crop}

        crop_objs += [crop_obj]

    return crop_objs  # [{'filename':filename,'data':data},{},{}]


def get_crop_objs_from_train(train_dir, annotation_json_path):
    print('>>>get_crop_objs_from_train...')
    df = pd.read_json(annotation_json_path)

    # mem may boom in this for

    max_step = 100
    step = 0
    pbar = tqdm(total=max_step,position=0, leave=True)

    cnt = 0
    crop_objs = []
    for filename in os.listdir(train_dir):
        if filename.endswith(".jpg"):
            file_full_path = os.path.join(train_dir, filename)

            img = cv2.imread(file_full_path)
            img_labels = df[df.name == filename]
            crop_objs_part = get_crop_objs_from_image(img, img_labels)

            crop_objs += crop_objs_part

            # break pieces to prevent mem BOOM
            pbar.update(1)
            step += 1
            cnt +=1
            # print('Counting objs in crop_objs: ' + str(cnt))
            if step == max_step:
                print('>>>max_step=' + str(max_step) + ' reached! ')
                print('>>>finished cnt:'+str(cnt))
                write_crops_to_disk(crop_objs, CROPS_DIR)
                crop_objs = []
                step = 0
                pbar.reset()

    pbar.close()

    return crop_objs


def write_crops_to_disk(crop_objs, dir_name):
    print('>>>write_crop_objs_to_disk...')

    for crop_obj in crop_objs:
        filename = crop_obj['filename']
        img = crop_obj['data']
        cv2.imwrite(dir_name + filename, img)

    return 1


def get_crops_from_val():
    pass


def get_result():
    pass


def make_submit():
    pass


def main():
    # train_dir   =   "ds/v4/big"
    train_dir = "ds/_origin/tile_round1_train_20201231/train_imgs/"
    annotation_json_path = "ds/v4/train_annos.json"


    ask_stale_imgs(CROPS_DIR)




    crop_objs = get_crop_objs_from_train(train_dir, annotation_json_path)
    # write_crops_to_disk(crop_objs, CROPS_DIR) # full crop objs will make mem BOOM!


if __name__ == "__main__":
    main()
