from v4 import *





def get_crop_objs_from_image2(img,filename,img_labels, crop_size=CROP_SIZE):
    img_width = img.size[0]
    img_height = img.size[1]

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

        # cv2 sln
        # crop = img[base_y:base_y + crop_size_y, base_x:base_x + crop_size_x]

        # PIL sln
        crop = img.crop((base_x, base_y, base_x + crop_size_x, base_y + crop_size_y))

        crop_bbox_x = bbox[0] - base_x + 1
        if bbox[0] -base_x <0:
            crop_bbox_x = 0

        crop_bbox_y = bbox[1] - base_y + 1
        if bbox[1]-base_y<0:
            crop_bbox_y=0

        crop_bbox_w = bbox[2] - base_x + 1
        if bbox[2] - base_x >= crop_size_x:
            crop_bbox_w = base_x+crop_size_x

        crop_bbox_h = bbox[2] - base_y + 1
        if bbox[3] - base_y >= crop_size_y:
            crop_bbox_h=base_y+crop_size_y

        crop_bbox = [crop_bbox_x, crop_bbox_y ,crop_bbox_w ,crop_bbox_h]
        # crop_bbox=[bbox[0]-base_x,bbox[1]-base_y,bbox[2]-base_x,bbox[3]-base_y]

        img_name_witout_ext = os.path.splitext(row['name'])[0]
        x, y, w, h = convert([crop_size_x, crop_size_y], crop_bbox)
        filename = '{0}__{1}__{2}__{3}__{4}__{5}__{6}__{7}__{8}__{9}.jpg'.format(
            img_name_witout_ext, base_x, base_y, crop_size_x, crop_size_y, label_index, x, y, w, h
        )  # alternate '__'.join(x,y,w,h)


        crop_obj = {'filename': filename, 'data': crop}
        crop_objs += [crop_obj]
    return crop_objs  # [{'filename':filename,'data':data},{},{}]


def get_crop_objs_from_train2(train_dir, annotation_json_path):
    print('>>>get_crop_objs_from_train...')
    df = pd.read_json(annotation_json_path)


    max_step = CROP_OBJS_MAXSTEP
    step = 0

    filenames= os.listdir(train_dir)
    cnt = 0
    crop_objs = []
    total = len(filenames)

    print('>>>making img_objs')
    img_objs={}
    for filename in filenames:
        if filename.endswith(".jpg"):
            file_full_path = os.path.join(train_dir, filename)
            img = Image.open(file_full_path)
            img_objs[filename]=img
    print('>>> img_objs ok')

    pbar = tqdm(total=max_step, position=0, leave=True)
    for filename, img in img_objs.items():
            img_labels = df[df.name == filename]
            crop_objs_part = get_crop_objs_from_image2(img,filename, img_labels)

            crop_objs += crop_objs_part
            img_objs[filename]=None # free mem

            # break pieces to prevent mem BOOM
            pbar.update(1)
            step += 1
            cnt += 1
            # print('Counting objs in crop_objs: ' + str(cnt))
            if step == max_step:
                print('>>>max_step=' + str(max_step) + ' reached! ')
                write_crops_to_disk(crop_objs, CROPS_DIR)
                print('>>>finished cnt:' + str(cnt) + '/' + str(total))
                crop_objs = []
                step = 0
                pbar.reset()

    # the last
    write_crops_to_disk(crop_objs, CROPS_DIR)
    print('>>>finished cnt:' + str(cnt) + '/' + str(total))
    pbar.close()

    return crop_objs





# step 1 cut flaw crops
get_crop_objs_from_train2(TRAIN_DIR, JSON_PATH)
exit()













# step 6 get result
tile_json_path = r'yolov5-master\runs\test\exp13\kaggle_1000_predictions.json'
final_json_path = 'ds/v4/final.json'
get_final_json(tile_json_path, final_json_path)
exit()







exit()
# img_size = [8192, 6000]
# bboxes = get_tiles(img_size,crop_size,lap_size)
# img = cv2.imread(r'ds\v4\big\197_2_t20201119084923676_CAM3.jpg')
# preview_tiles(img,bboxes)

# print(1)


exit()
# [
#     {
#         "name": "226_46_t20201125133518273_CAM1.jpg",
#         "category": 4,
#         "bbox": [
#             5662,
#             2489,
#             5671,
#             2497
#         ],
#         "score": 0.130576
#     },

def get_final_json(tile_json_path,final_json_path):

    json_obj=[]
    df = pd.read_json(tile_json_path)
    for index, row in df.iterrows():


        category = int(row['category_id'])
        ele = row['image_id'].split("__")


        filename = ele[0] + '.jpg'

        crop_x,crop_y,crop_w,crop_h = row['bbox']
        base_x,base_y=int(ele[1]),int(ele[2])
        x,y,r,b = base_x + crop_x, base_y+crop_y,base_x + crop_x +crop_w, base_y + crop_y + crop_h
        score = row['score']


        flaw_obj = {
                "name": filename,
                "category": category,
                "bbox": [
                    round(x, 2),
                    round(y, 2),
                    round(r, 2),
                    round(b, 2)
                ],
                "score": score
        }

        # print(flaw_obj)

        json_obj += [flaw_obj]

    print(json_obj)

    # exit()

    with open(final_json_path, 'w') as fp:
        print('>>> wrighting finnal json')
        json.dump(json_obj, fp, indent=4, ensure_ascii=False)

    # print(df)

tile_json_path = r'yolov5-master\runs\test\exp11\kaggle_1000_predictions.json'
final_json_path = 'ds/v4/final.json'
get_final_json(tile_json_path, final_json_path)


exit()


list1 = []
cnt = 0
for filename in os.listdir(TEST_DIR):
    # img = cv2.imread(TEST_DIR + filename)
    # list1 += [img.shape[0]]

    im = Image.open(TEST_DIR + filename)
    list1 += [im.size[1]]

    cnt += 1
    print(cnt)

output = set()
for x in list1:
    output.add(x)
print(output)



exit()
for i in range(3):
    print(i)

print()

exit()

from v4 import *

print('>>>check_unique_vals(JSON_PATH)')
check_unique_vals(JSON_PATH)
