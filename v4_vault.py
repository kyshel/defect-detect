from v4 import *


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
