#!/usr/bin/env python
# -*- coding: utf-8 -*-
from step1_lib1 import *




image = cv2.imread(r'D:\ml\p8\ds\step1\sample10\hang\197_1_t20201119084916148_CAM1.jpg')
# ----------------




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

def main():
    train10_dir = "ds/step1/train10/"
    json_path = "ds/step1/train_annos.json"
    LABEL_PATH = 'tdd\\labels\\train2017\\'
    df = pd.read_json(json_path)
    df_merged = df.groupby('name', as_index=False).agg(lambda x: x.tolist())
    #
    # print(df_merged)




    cnt = 0
    txt_dict = {}
    for index, row in df_merged.iterrows():
        txt_filename = os.path.splitext( row['name'])[0]
        line_list=[]
        for i in range(len(row['image_width'])):
            label_class = row['category'][i]
            shape = [row['image_width'][i],row['image_height'][i]]
            bbox = row['bbox'][i]
            x,y,w,h = convert(shape,bbox)
            line = [label_class,x,y,w,h]
            line_list += [line]
        txt_dict[txt_filename]=line_list

    print(json.dumps(txt_dict, indent=4, sort_keys=True))


    # txt_dict = {'1.jpg': [[class,x,y,w,h],[class,x,w,y,h]],'2.jpg':[...]}
    for filename,line_list in txt_dict.items():
        with open(LABEL_PATH+filename + ".txt", "w") as text_file:
            for line in line_list:
                text_file.write("{} {} {} {} {}\n".format(line[0],line[1],line[2],line[3],line[4]))


    exit()


















if __name__ == "__main__":
    main()



