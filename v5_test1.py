from v5 import *




def write_txts_obj_to_disk(txts_obj,txt_dir):
    print('>>>write_txts_obj_to_disk...')

    # tqdm
    max_step = len(txts_obj)
    pbar = tqdm(total=max_step, position=0, leave=True)

    cnt=0
    for img_name,lines in txts_obj.items():
        txt_filename = os.path.splitext(img_name)[0] + ".txt"
        with open(txt_dir+txt_filename, "w") as text_file:
            for line in lines:
                text_file.write("{} {} {} {} {}\n".format(line[0],line[1],line[2],line[3],line[4]))
        cnt+=1

        pbar.update(1)
    pbar.close()

    print('>>>txt files cnt:'+str(cnt)+'. making done,check  ' + txt_dir)
    print()




def json_to_txt(json_path,txt_dir):
    df = pd.read_json(json_path)
    squashed=df.groupby('name', as_index=False).agg(lambda x: x.tolist())

    txts_obj={} # {'img1':[[class,cx,cy,w,h],[class,cx,cy,w,h]],'img2':...}
    for index, row in squashed.iterrows():

        img_name=row['name']
        bboxes = row['bbox']
        img_w = row['image_width'][0]
        img_h=row['image_height'][0]
        shape=(img_w,img_h)

        txt_lines = []
        for i,bbox in enumerate(bboxes):
            cx,cy,w,h = convert(shape,bbox)
            img_class = row['category'][i]
            txt_line= [img_class]
            txt_line+= cx,cy,w,h
            txt_lines+=[txt_line]

        txts_obj[img_name] = txt_lines

    write_txts_obj_to_disk(txts_obj,txt_dir)







json_to_txt(JSON_PATH, BIG_TXT_DIR)







