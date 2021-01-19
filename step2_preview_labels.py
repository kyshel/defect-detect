#!/usr/bin/env python
# -*- coding: utf-8 -*-
from step1_lib1 import *



def draw_img(df, dir_name, filename):
    file_full_path = os.path.join(dir_name, filename)
    img_labels = df[df.name == filename]
    logging.info(img_labels)
    img = cv2.imread(file_full_path)

    roi_list = []
    for index, row in img_labels.iterrows():
        category = row['category']
        color = flaw(category)['color']
        bbox = row['bbox']

        w = int(bbox[2] - bbox[0])
        h = int(bbox[3] - bbox[1])
        x = int(bbox[0])
        y = int(bbox[1])

        off = 100
        rec_left_upper_x = int(x - off)
        rec_left_upper_y = int(y - off)
        draw_cross(img, (x, y))
        cv2.rectangle(img, (x, y, w, h), color, 1)
        cv2.rectangle(img, (rec_left_upper_x, rec_left_upper_y,
                            w + 2 * off, h + 2 * off), color, 10)
        text ="{},x{},y{},w{},h{}".format(flaw(category)['name'],x,y,w,h)
        cv2.putText(img, text, (x - 40, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return img





def get_roi_list(df, dir_name, filename):
    file_full_path = os.path.join(dir_name, filename)
    img_labels = df[df.name == filename]
    logging.info(img_labels)
    img = cv2.imread(file_full_path)

    roi_list = []
    for index, row in img_labels.iterrows():
        category = row['category']
        color = flaw_color(category)
        bbox = row['bbox']

        w = int(bbox[2] - bbox[0])
        h = int(bbox[3] - bbox[1])
        x = int(bbox[0])
        y = int(bbox[1])



        off = 100
        rec_left_upper_x = int(x - off)
        rec_left_upper_y = int(y - off)
        draw_cross(img, (x, y))
        cv2.rectangle(img, (x, y, w, h), color, 1)
        cv2.rectangle(img, (rec_left_upper_x, rec_left_upper_y,
                            w + 2 * off, h + 2 * off), color, 10)
        text = "{},x{},y{},w{},h{}".format(flaw(category)['name'], x, y, w, h)
        cv2.putText(img, text, (x - 40, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)



        roi = img[rec_left_upper_y:rec_left_upper_y + h + 2 * off,
              rec_left_upper_x:rec_left_upper_x + w + 2 * off]
        roi_list += [roi]

    return roi_list



FLAW_CROP_SIZE = 512
OUT_DIR = 'ds/step2/output/'
CROPS_DUR = 'ds/step2/crops/'
CROPS_TRAIN_PATH = ''



###   ----------          rebuild v3              ----------------

def work_flow():
    crops_train = get_crops_from_train()
    write_crops_to_disk(crops_train, CROPS_TRAIN_PATH)

    crops_val = get_crops_from_val()
    result = get_result(crops_val)

    make_submit(result)



def get_crop_objs_from_image(img,img_labels,crop_size = CROP_SIZE):
    img_width=img.shape[1]
    img_height=img.shape[0]
    crop_objs = []
    for index, row in img_labels.iterrows():
        label_index = row['category']
        bbox = row['bbox']

        cx = (bbox[2] + bbox[0])/2
        cy = (bbox[3] + bbox[1]) / 2
        base_x = int(cx - crop_size / 2)
        base_y = int(cy - crop_size / 2)

        crop_size_x, crop_size_y=crop_size,crop_size
        if base_x < 0 : base_x = 0
        if base_y < 0 : base_y = 0
        if base_x + crop_size > img_width: crop_size_x = img_width - base_x
        if base_y + crop_size > img_height: crop_size_y = img_height - base_y

        crop = img[base_y:base_y+crop_size_y, base_x:base_x+crop_size_x]
        crop_bbox=[bbox[0]-base_x,bbox[1]-base_y,bbox[2]-base_x,bbox[3]-base_y]


        x,y,w,h=convert([crop_size_x,crop_size_y],crop_bbox)
        filename = '{0}__{1}__{2}__{3}__{4}__{5}__{6}__{7}__{8}__{9}.jpg'.format(
            row['name'], base_x, base_y, crop_size_x, crop_size_y, label_index, x, y, w, h
        )  # alternate '__'.join(x,y,w,h)


        # draw some graph
        x = int(crop_bbox[0])
        y = int(crop_bbox[1])
        w = int(crop_bbox[2] - crop_bbox[0])
        h = int(crop_bbox[3] - crop_bbox[1])

        draw_cross(crop, (x, y))
        color = flaw(label_index)['color']
        cv2.rectangle(crop, (x, y, w, h), color, 1)
        cv2.rectangle(crop, (0, 0,
                            crop_size_x, crop_size_y), color, 10)
        text = "{},x{},y{},w{},h{}".format(flaw(label_index)['name'], x, y, w, h)
        cv2.putText(crop, text, (x - 40, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # build flaw_crop
        crop_obj = {'filename': filename, 'data': crop}

        print('crop_obj>>>')
        print(crop_obj)
        print('crop_obj<<<')

        crop_objs += [crop_obj]

    return crop_objs #  [{'filename':filename,'data':data},{},{}]



def get_crop_objs_from_train(train_dir, annotation_json_path):
    df = pd.read_json(annotation_json_path)

    # mem may boom in this for
    crop_objs=[]
    for filename in os.listdir(train_dir):
        if filename.endswith(".jpg"):
            file_full_path = os.path.join(train_dir, filename)

            img = cv2.imread(file_full_path)
            img_labels = df[df.name == filename]
            crop_objs_part = get_crop_objs_from_image(img,img_labels)

            print('crop_objs_part>>>')
            print(crop_objs_part)
            print('crop_objs_part<<<')

            crop_objs += crop_objs_part

    return crop_objs


def write_crops_to_disk(crop_objs,dir_name):
    # clean output dir
    for item in os.listdir(dir_name):
        if item.endswith(".jpg"):
            os.remove(os.path.join(dir_name, item))

    for crop_obj in crop_objs:
        filename = crop_obj['filename']
        print(filename)
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
    train_dir = "ds/step2/train10/"
    annotation_json_path = "ds/step2/train_annos.json"
    crops_train_dir = "ds/step2/crops/"



    crop_objs = get_crop_objs_from_train(train_dir, annotation_json_path)
    print('crop_objs>>>')
    print(crop_objs)
    print('crop_objs<<<')
    write_crops_to_disk(crop_objs, crops_train_dir)




    exit()





    train10_dir = "ds/step2/train10/"
    json_path = "ds/step2/train_annos.json"

    df = pd.read_json(json_path)

    # clean output dir
    for item in os.listdir(OUT_DIR):
        if item.endswith(".jpg"):
            os.remove(os.path.join(OUT_DIR, item))


    # write drawed img
    img_list=[]
    for filename in os.listdir(train10_dir):
        if filename.endswith(".jpg"):
            # draw_img2(df, train10_dir, filename)
            img = draw_img(df, train10_dir, filename)
            img_list += [img]
    write_imgs_to_disk(img_list, OUT_DIR)

    # write crops
    crops = []
    for filename in os.listdir(train10_dir):
        if filename.endswith(".jpg"):
            # draw_img2(df, train10_dir, filename)
            roi_list = get_roi_list(df, train10_dir, filename)
            crops += roi_list

            # break

    write_imgs_to_disk(crops, CROPS_DUR)




if __name__ == "__main__":
    main()
