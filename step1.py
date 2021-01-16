#!/usr/bin/env python
# -*- coding: utf-8 -*-
from step1_primary import *








# cv2.imwrite(OUT_DIR + row['name'], img)


































def main():
    train10_dir = "ds/step1/train10"
    json_path = "ds/step1/train_annos.json"
    var_dump(train10_dir)

    df = pd.read_json(json_path)

    for filename in os.listdir(train10_dir):
        if filename.endswith(".jpg"):
            draw_img2(df, train10_dir, filename)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
