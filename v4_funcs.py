#!/usr/bin/env python
# -*- coding: utf-8 -*-

from v4_imports_and_constants import *



def ins(v):
    print("ins>>>")
    print('>dir:')
    print(dir(v))
    print('>type:')
    print(type(v))
    print('>print:')
    print(v)
    print("ins<<<")







def get_box_resized(origin, ratio):
    box = origin.copy()
    for i in range(len(box)):
        box[i] = int(box[i] * ratio)
    return box


def get_shape_resized(shape, ratio):
    width = int(shape[1] * ratio)
    height = int(shape[0] * ratio)
    return width, height


def get_cutted_filename(name):
    part = name.split('_')
    return part[0] + '_' + part[1] + '_' + part[3]


def random_color():
    color = np.random.randint(0, 255, size=(3,))
    return int(color[0]), int(color[1]), int(color[2])

def flaw(index):
    flaw_info ={
        0:{
            'name':'0 background',
            'name_cn':'0背景',
            'color':[255, 165, 0],
        },
        1: {
            'name': '1 edge weird',
            'name_cn': '1边异常',
            'color': GREEN,
        },
        2: {
            'name': '2 corner weird',
            'name_cn': '2角异常',
            'color': BLUE,
        },
        3: {
            'name': '3 white spot',
            'name_cn': '3白色点瑕疵',
            'color': RED,
        },
        4: {
            'name': '4 light block',
            'name_cn': '4浅色块瑕疵',
            'color': YELLOW,
        },
        5: {
            'name': '5 dark block',
            'name_cn': '5深色点块瑕疵',
            'color': PINK,
        },
        6: {
            'name': '6 halo',
            'name_cn': '6光圈瑕疵',
            'color': PURPLE,
        },
    }
    return flaw_info[index]


def flaw_name(num):
    map_type = {
        0: "0背景",
        1: "1边异常",
        2: "2角异常",
        3: "3白色点瑕疵",
        4: "4浅色块瑕疵",
        5: "5深色点块瑕疵",
        6: "6光圈瑕疵"
    }
    map_type1 = {
        0: "0 background",
        1: "1 edge weird",
        2: "2 corner weird",
        3: "3 white spot",
        4: "4 light block",
        5: "5 dark block",
        6: "6 halo"
    }
    return map_type1[num]


def flaw_color(num):
    map_color = {
        0: [255, 165, 0],
        1: GREEN,
        2: [255, 255, 0],
        3: RED,
        4: [0, 0, 255],
        5: [75, 0, 130],
        6: [238, 130, 238]
    }
    return map_color[num]


def convertToRelativeValues(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    cx = (box[1] + box[0]) / 2.0
    cy = (box[3] + box[2]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = cx * dw
    y = cy * dh
    w = w * dw
    h = h * dh
    # x,y => (bounding_box_center)/width_of_the_image
    # w => bounding_box_width / width_of_the_image
    # h => bounding_box_height / height_of_the_image
    return (x, y, w, h)


def convertToAbsoluteValues(size, box):
    # w_box = round(size[0] * box[2])
    # h_box = round(size[1] * box[3])
    xIn = round(((2 * float(box[0]) - float(box[2])) * size[0] / 2))
    yIn = round(((2 * float(box[1]) - float(box[3])) * size[1] / 2))
    xEnd = xIn + round(float(box[2]) * size[0])
    yEnd = yIn + round(float(box[3]) * size[1])
    if xIn < 0:
        xIn = 0
    if yIn < 0:
        yIn = 0
    if xEnd >= size[0]:
        xEnd = size[0] - 1
    if yEnd >= size[1]:
        yEnd = size[1] - 1
    return (xIn, yIn, xEnd, yEnd)


def aaa(t):
    x = (t['bbox'][0] + t['bbox'][2]) / 2 / t['image_width']
    y = (t['bbox'][1] + t['bbox'][3]) / 2 / t['image_height']
    w = (t['bbox'][2] - t['bbox'][0]) / t['image_width']
    h = (t['bbox'][3] - t['bbox'][1]) / t['image_height']
    return x, y, w, h


def xywh(size, box):
    x = box[0]
    y = box[1]
    w = box[2] - box[0]
    h = box[3] - box[1]
    return x, y, w, h


def draw_img(df, dir_name, filename):
    file_full_path = os.path.join(dir_name, filename)
    img_labels = df[df.name == filename]
    logging.info(img_labels)
    resize_ratio = 1

    img = cv2.imread(file_full_path)
    shape_resized = get_shape_resized(img.shape, resize_ratio)
    img_resized = cv2.resize(img, shape_resized)
    print(img.shape)
    print(img_resized.shape)

    i = 0
    for index, row in img_labels.iterrows():
        bbox_resized = get_box_resized(row['bbox'], resize_ratio)
        print(row['bbox'])
        print(bbox_resized)

        x, y, w, h = xywh(shape_resized, bbox_resized)
        category = row['category']
        color = flaw_color(category)

        cv2.rectangle(img_resized, (x, y, w, h), color, 1)
        cv2.putText(img_resized, flaw_name(category), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # cv2.imwrite(OUT_DIR + row['name'], img_resized)

    # cv2.imshow(get_cutted_filename(filename) + '_' + str(i), img_resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # exit()


def draw_cross(img, loc, color=BLUE, line_length=10):
    (x, y) = loc
    cv2.line(img, (x, y - line_length), (x, y + line_length), color, 1)
    cv2.line(img, (x - line_length, y), (x + line_length, y), color, 1)


# 85 ,70 ,90
THRESHOLD_OF_BINARY = 90
# 10
CONTOUR_MINIMAL_AREA = 10
# 420
CENTER_CIRCULE_RADIUS = 205
# 0
DEBUG_IMSHOW = 0

import time

START_TIME = time.time()


def hit():
    print("hitted --- %s seconds ---" % (time.time() - START_TIME))



# stale
def get_region(img):
    ## Threshold in grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray_three = cv2.merge([gray, gray, gray])


    # return gray_three



    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
    titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

    exit()

    retval, threshed = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    ## Find wathc region by counting the projector
    h, w = img.shape[:2]
    x = np.sum(threshed, axis=0)
    y = np.sum(threshed, axis=1)
    yy = np.nonzero(y > (w / 5 * 255))[0]
    xx = np.nonzero(x > (h / 5 * 255))[0]
    region = img[yy[0]:yy[-1], xx[0]:xx[-1]]
    cv2.imshow("region.png" + get_date(), region) if DEBUG_IMSHOW else 1

    return region




def get_date():
	return str(int(round(time.time() * 1000)))[-4:]

DEBUG_IMSHOW = 1


def see(img):
    see1(img)


def see0():
    see1(img)
    see2(img)
    see3(img)

# PIL call sys
def see1(img):
    print('>>> see1, or see, PIL call sys ')
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img2)
    im_pil.show()

# matploit
def see2(img):
    print('>>> see2 ,matploit ')
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(RGB_img)
    plt.xticks([]), plt.yticks([])
    plt.show()

# opencv
def see3(img):
    print('>>> see3 ,opencv ')
    cv2.imshow( get_date() , img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_date():
	return str(int(round(time.time() * 1000)))[-4:]

def get_contours(region):
    ## Change to LAB space
    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    imglab = np.hstack((l, a, b))
    plt.imshow(imglab)
    plt.xticks([]), plt.yticks([])
    plt.show()
    exit()

    cv2.imshow("region_lab.png" + get_date(), imglab) if DEBUG_IMSHOW else 1

    ## normalized the a channel to all dynamic range
    na = cv2.normalize(a, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    cv2.imshow("a_normalized.png" + get_date(), na) if DEBUG_IMSHOW else 1

    ## Threshold to binary
    retval, threshed = cv2.threshold(na, thresh=THRESHOLD_OF_BINARY, maxval=255, type=cv2.THRESH_BINARY)

    ## Do morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, kernel)
    res = np.hstack((threshed, opened))
    cv2.imshow("a_binary.png" + get_date(), res) if DEBUG_IMSHOW else 1

    ## Find contours
    contours = cv2.findContours(opened, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)[-2]

    return contours



def get_img_list_from_disk(dir_name):
    crops = []
    for filename in os.listdir(dir_name):
        img = cv2.imread(dir_name + filename)
        crops += [img]
    return crops

# convert bbox to yolo txt
def convert(shape, bbox):
    dw = 1./(shape[0])
    dh = 1./(shape[1])
    x = (bbox[0] + bbox[2])/2.0 - 1
    y = (bbox[1] + bbox[3])/2.0 - 1
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def write_imgs_to_disk(img_list,dir_name):
    for i, x in enumerate(img_list):
        cv2.imwrite(dir_name + str(i) + ".jpg", x)
    return 1

import sys

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")




def ask_stale_imgs(dir_name = CROPS_DIR):
    if query_yes_no('Reserve stale imgs in ' + dir_name + ' ? ', default="yes"):
        print('Stale img reserved.')
    else:
        print('Cleaning stale imgs (ending with .jpg)...')
        for item in os.listdir(dir_name):
            if item.endswith(".jpg"):
                os.remove(os.path.join(dir_name, item))



def check_unique_vals(json_file):
    df = pd.read_json(json_file)
    print(df['image_width'].unique())

def get_tiles(img_size,crop_size,lap_size):
    img_w,img_h=img_size[1],img_size[0]
    step = crop_size - lap_size
    cnt_x =  int(img_w / step)
    cnt_y =  int(img_h / step)


    # loop y ,  complete
    bboxes=[]
    for cur_y in range(cnt_y):
        base_y = cur_y * step
        crop_size_x, crop_size_y = crop_size, crop_size
        for cur_x in range(cnt_x):
            base_x = cur_x * step
            x,y,w,h = base_x,base_y,crop_size_x,crop_size_y
            bboxes += [[x,y,w,h]]

        # last col , not complete
        base_x = (cur_x + 1) * step
        if (base_x + crop_size_x) >= img_w:
            crop_size_x = img_w - base_x
            if crop_size_x != 0:
                x,y,w,h = base_x,base_y,crop_size_x,crop_size_y
                bboxes += [[x, y, w, h]]



    # last row, not complete
    base_y = (cur_y + 1) * step
    crop_size_x, crop_size_y = crop_size, crop_size
    if (base_y + crop_size_y) >= img_h:
        crop_size_y = img_h - base_y
        if crop_size_y != 0:

            for cur_x in range(cnt_x):
                base_x = cur_x * step
                x, y, w, h = base_x, base_y, crop_size_x, crop_size_y
                bboxes += [[x, y, w, h]]


            # the Bottom right corner, not complete
            base_x = (cur_x + 1) * step
            if (base_x + crop_size_x) >= img_w:
                crop_size_x = img_w - base_x
                if crop_size_x != 0:
                    x, y, w, h = base_x, base_y, crop_size_x, crop_size_y
                    bboxes += [[x, y, w, h]]

    return  bboxes


def preview_tiles(img,bboxes):

    img_w,img_h = img.shape[1],img.shape[0]
    print(bboxes)
    for bbox in bboxes:
        color = random_color()
        color = WHITE
        x,y,w,h=bbox[0],bbox[1],bbox[2],bbox[3]
        cv2.rectangle(img, (x, y, w,h), color, 1)
    see(img)


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




def make_txt_from_crops(dir_crops,dir_txt,val_txt_dir = VAL_TXT_DIR):
    print('[]make_txt_from_crops')
    # clean dir
    for item in os.listdir(dir_txt):
        if item.endswith(".txt"):
            os.remove(os.path.join(dir_txt, item))

    txt_dict = {}
    for filename in os.listdir(dir_crops):
        if filename.endswith(".jpg"):
            txt_filename = os.path.splitext(filename)[0]
            ele = txt_filename.split('__')
            txt_line = [ele[5],ele[6],ele[7],ele[8],ele[9]]
            txt_dict[txt_filename] = txt_line

    print('>>>writing txt to '+dir_txt)
    cnt=0
    for filename,line in txt_dict.items():
        with open(dir_txt+filename + ".txt", "w") as text_file:
            text_file.write("{} {} {} {} {}\n".format(line[0],line[1],line[2],line[3],line[4]))
        cnt+=1
    print('txt files cnt:'+str(cnt))
    print('making done,check  ' + dir_txt)


    print('>>>writing txt to ' +val_txt_dir)
    cnt = 0
    for filename, line in txt_dict.items():
        with open(val_txt_dir + filename + ".txt", "w") as text_file:
            text_file.write("{} {} {} {} {}\n".format(line[0], line[1], line[2], line[3], line[4]))
        cnt += 1
    print('txt files cnt:' + str(cnt))
    print('making done,check  '+val_txt_dir)





def get_crop_objs_from_image(file_full_path, img_name,img_labels, crop_size=CROP_SIZE):
    # cv2 sln
    # img = cv2.imread(file_full_path)
    # img_width = img.shape[1]
    # img_height = img.shape[0]

    # PIL sln
    img = Image.open(file_full_path)
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
    pbar = tqdm(total=max_step, position=0, leave=True)

    filenames= os.listdir(train_dir)
    cnt = 0
    crop_objs = []
    total = len(filenames)
    for filename in filenames:
        if filename.endswith(".jpg"):
            file_full_path = os.path.join(train_dir, filename)

            img_labels = df[df.name == filename]
            crop_objs_part = get_crop_objs_from_image(file_full_path,filename, img_labels)

            crop_objs += crop_objs_part

            # break pieces to prevent mem BOOM
            pbar.update(1)
            step += 1
            cnt += 1
            # print('Counting objs in crop_objs: ' + str(cnt))
            if step == max_step:
                pbar.reset()
                print('>>>max_step=' + str(max_step) + ' reached! ')
                write_crops_to_disk(crop_objs, CROPS_DIR)
                print('>>>finished cnt:' + str(cnt) + '/' + str(total))
                crop_objs = []
                step = 0


    # the last
    write_crops_to_disk(crop_objs, CROPS_DIR)
    pbar.close()
    print('>>>finished cnt:' + str(cnt) + '/' + str(total))


    return crop_objs


def write_crops_to_disk(crop_objs, dir_name=CROPS_DIR):
    print('>>>write_crop_objs_to_disk...')

    for crop_obj in crop_objs:
        filename = crop_obj['filename']
        img = crop_obj['data']
        # cv2.imwrite(dir_name + filename, img)
        img.save(dir_name + filename, "JPEG")

    return 1

# rm
def write_tiles_to_disk(tile_objs, dir_name=TILES_DIR):
    print('>>>write_tiles_to_disk...')

    for tile_obj in tile_objs:
        filename = tile_obj['filename']
        img = tile_obj['data']
        img.save(dir_name+filename, "JPEG")

    return 1


def get_tile_objs_from_image(file_full_path, img_name, crop_holes):
    # cv2 sln
    # img = cv2.imread(file_full_path)
    # img_width = img.shape[1]
    # img_height = img.shape[0]

    # PIL sln
    img = Image.open(file_full_path)
    img_width = img.size[0]
    img_height = img.size[1]

    if img_width == 8192:
        tiles = crop_holes[0]
    elif img_width == 4096:
        tiles = crop_holes[1]
    else:
        print('img size nor 8192 neither 4096, stopped.')
        exit()

    img_name_witout_ext = os.path.splitext(img_name)[0]
    tile_objs = []
    for index, bbox in enumerate(tiles):
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        #tile = img[x:x + w, y:y + h]
        tile = img.crop((x, y, x+w, y+h))
        filename = '{0}__{1}__{2}__{3}__{4}.jpg'.format(
            img_name_witout_ext, x, y, w, h
        )  # alternate '__'.join(x,y,w,h)
        tile_obj = {'filename': filename, 'data': tile}

        tile_objs += [tile_obj]

    # print(tile_objs)

    return tile_objs  # [{'filename':filename,'data':data},{},{}]


def get_tile_objs_from_test(test_dir, crop_holes):
    print('>>>get_tile_objs_from_test...')

    # mem may boom in this for
    max_step = TILE_OBJS_MAXSTEP
    step = 0
    pbar = tqdm(total=max_step, position=0, leave=True)

    cnt = 0
    tile_objs = []
    filenames = os.listdir(test_dir)
    total = len(filenames)
    for filename in filenames:
        if filename.endswith(".jpg"):
            file_full_path = os.path.join(test_dir, filename)


            # img_labels = df[df.name == filename]
            tile_objs_part = get_tile_objs_from_image(file_full_path, filename, crop_holes)
            # print(tile_objs_part)

            tile_objs += tile_objs_part

            # break pieces to prevent mem BOOM
            pbar.update(1)
            step += 1
            cnt += 1
            # print('Counting objs in tile_objs: ' + str(cnt))
            if step == max_step:
                print('>>>max_step=' + str(max_step) + ' reached! ')
                write_crops_to_disk(tile_objs, TILES_DIR)
                print('>>>finished cnt:' + str(cnt) + '/' + str(total))
                tile_objs = []
                step = 0
                pbar.reset()

    # the last
    write_crops_to_disk(tile_objs, TILES_DIR)
    print('>>>finished cnt:' + str(cnt) + '/' + str(total))
    pbar.close()

    # print(tile_objs)

    return tile_objs




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

        crop_bbox_h = bbox[3] - base_y + 1
        if bbox[3] - base_y >= crop_size_y:
            crop_bbox_h=base_y+crop_size_y

        crop_bbox = [crop_bbox_x, crop_bbox_y ,crop_bbox_w ,crop_bbox_h]
        #print(crop_bbox)
        #exit()
        # crop_bbox=[bbox[0]-base_x,bbox[1]-base_y,bbox[2]-base_x,bbox[3]-base_y]

        img_name_witout_ext = os.path.splitext(row['name'])[0]
        x, y, w, h = convert([crop_size_x, crop_size_y], crop_bbox)
        filename = '{0}__{1}__{2}__{3}__{4}__{5}__{6}__{7}__{8}__{9}.jpg'.format(
            img_name_witout_ext, base_x, base_y, crop_size_x, crop_size_y, label_index, x, y, w, h
        )  # alternate '__'.join(x,y,w,h)

        # # draw some graph
        # x = int(crop_bbox[0])
        # y = int(crop_bbox[1])
        # w = int(crop_bbox[2] - crop_bbox[0])
        # h = int(crop_bbox[3] - crop_bbox[1])
        # from PIL import Image, ImageDraw
        # img1 = ImageDraw.Draw(crop)
        # img1.rectangle((x,y,x+w,y+h), outline="red")
        # # crop.show()

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



