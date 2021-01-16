#!/usr/bin/env python
# -*- coding: utf-8 -*-
from step1_lib1 import *


def draw_img2(df, dir_name, filename):
    file_full_path = os.path.join(dir_name, filename)
    print(file_full_path)
    img_labels = df[df.name == filename]
    logging.info(img_labels)
    img = cv2.imread(file_full_path)

    # img = get_region(img)

    i = 0

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
        rec_left_upper_x = int(x - (w + off) / 2)
        rec_left_upper_y = int(y - (h + off) / 2)
        draw_cross(img, (x, y))
        cv2.rectangle(img, (x, y, w, h), color, 1)
        cv2.rectangle(img, (rec_left_upper_x, rec_left_upper_y, w + off, h + off), color, 5)
        cv2.putText(img, flaw_name(category), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        roi = img[rec_left_upper_y:rec_left_upper_y + h + off,
              rec_left_upper_x:rec_left_upper_x + w + off
              ]

        roi_list += [roi]

    axes = []
    for i, x in enumerate(roi_list):
        axes.append(FIG.add_subplot(ROWS, COLS, i + 1))
        subplot_title = ("roi" + str(i))
        axes[-1].set_title(subplot_title)
        plt.imshow(x)


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

        # off = 100
        # rec_left_upper_x = int(x - (w + off) / 2)
        # rec_left_upper_y = int(y - (h + off) / 2)
        # draw_cross(img, (x, y))
        # cv2.rectangle(img, (x, y, w, h), color, 1)
        # cv2.rectangle(img, (rec_left_upper_x, rec_left_upper_y, w + off, h + off), color, 5)
        # cv2.putText(img, flaw_name(category), (x-40, y - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        off = 50
        rec_left_upper_x = int(x - off)
        rec_left_upper_y = int(y - off)
        draw_cross(img, (x, y))
        cv2.rectangle(img, (x, y, w, h), color, 1)
        cv2.rectangle(img, (rec_left_upper_x, rec_left_upper_y,
                            w + 2 * off, h + 2 * off), color, 5)
        cv2.putText(img, flaw_name(category), (x - 40, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)



        roi = img[rec_left_upper_y:rec_left_upper_y + h + 2 * off,
              rec_left_upper_x:rec_left_upper_x + w + 2 * off]
        roi_list += [roi]



    return roi_list


def write_crops_to_disk(crops, dir_name):
    for i, x in enumerate(crops):
        cv2.imwrite(dir_name + str(i) + ".jpg", x)
    return 1


def get_crops_from_disk(dir_name):
    crops = []
    for filename in os.listdir(dir_name):
        img = cv2.imread(dir_name + filename)
        crops += [img]
    return crops


def main():
    train10_dir = "ds/step1/train10/"
    json_path = "ds/step1/train_annos.json"
    df = pd.read_json(json_path)

    # temp empty
    for item in os.listdir(OUT_DIR):
        if item.endswith(".jpg"):
            os.remove(os.path.join(OUT_DIR, item))

    # get crops
    crops = get_crops_from_disk(OUT_DIR)
    if not crops:
        for filename in os.listdir(train10_dir):
            if filename.endswith(".jpg"):
                # draw_img2(df, train10_dir, filename)
                roi_list = get_roi_list(df, train10_dir, filename)
                crops += roi_list

                # break

        write_crops_to_disk(crops, OUT_DIR)

    # draw crops
    fig = plt.figure()
    axes = []
    rows = int(math.sqrt(len(crops))) + 1
    for i, x in enumerate(crops):
        axes.append(fig.add_subplot(rows, rows, i + 1))
        subplot_title = ("crop_" + str(i))
        axes[-1].set_title(subplot_title)
        plt.imshow(x)

    fig.tight_layout()
    plt.show()

    # hang
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
