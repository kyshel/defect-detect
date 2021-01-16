

from glob import glob

import matplotlib.patches as patches
import seaborn as sns
from bokeh.plotting import figure
from bokeh.io import output_notebook, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, Panel
from bokeh.models.widgets import Tabs
import albumentations as albu




TRAIN_DIR = 'ds/step1/train10/'
TEST_DIR = 'ds/_origin/tile_round1_testA_20201231/testA_imgs/'
TRAIN_CSV_PATH = 'ds/step1/train_annos.json'

# Glob the directories and get the lists of train and test images
train_fns = glob(TRAIN_DIR + '*')
test_fns = glob(TEST_DIR + '*')
print('Number of train images is {}'.format(len(train_fns)))
print('Number of test images is {}'.format(len(test_fns)))

import json
train = []
with open(TRAIN_CSV_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)
train = pd.DataFrame(data)

# print(train.head(5))


all_train_images = pd.DataFrame([fns.split('\\')[-1] for fns in train_fns])
all_train_images.columns=['name']

# merge image with json info
all_train_images = all_train_images.merge(train, on='name', how='left')
print(all_train_images)

# replace nan values with zeros
all_train_images['bbox'] = all_train_images.bbox.fillna('[0,0,0,0]')
all_train_images.head(5)

# [xmin，ymin，xmax，ymax]
bbox_items = all_train_images.bbox
all_train_images['bbox_xmin'] = bbox_items.apply(lambda x: x[0])
all_train_images['bbox_ymin'] = bbox_items.apply(lambda x: x[1])
all_train_images['bbox_width'] = bbox_items.apply(lambda x: x[2]-x[0])
all_train_images['bbox_height'] = bbox_items.apply(lambda x: x[3]-x[1])
# print(all_train_images)
print('{} images without bbox.'.format(len(all_train_images) - len(train)))



def get_all_bboxes(df, name):
    image_bboxes = df[df.name == name]

    bboxes = []
    for _, row in image_bboxes.iterrows():
        bboxes.append((row.bbox_xmin, row.bbox_ymin, row.bbox_width, row.bbox_height))

    return bboxes


def plot_image_examples(df, rows=3, cols=3, title='Image examples'):
    fig, axs = plt.subplots(rows, cols, figsize=(10, 10))
    for row in range(rows):
        for col in range(cols):
            idx = np.random.randint(len(df), size=1)[0]
            name = df.iloc[idx]["name"]
            img = Image.open(TRAIN_DIR + str(name))
            axs[row, col].imshow(img)

            bboxes = get_all_bboxes(df, name)

            for bbox in bboxes:
                rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r',
                                         facecolor='none')
                axs[row, col].add_patch(rect)

            axs[row, col].axis('off')

    plt.suptitle(title)


# plot_image_examples(all_train_images)

def hist_hover(dataframe, column, colors=["#94c8d8", "#ea5e51"], bins=30, title=''):
    hist, edges = np.histogram(dataframe[column], bins=bins)

    hist_df = pd.DataFrame({column: hist,
                            "left": edges[:-1],
                            "right": edges[1:]})
    hist_df["interval"] = ["%d to %d" % (left, right) for left,
                                                          right in zip(hist_df["left"], hist_df["right"])]

    src = ColumnDataSource(hist_df)
    plot = figure(plot_height=400, plot_width=600,
                  title=title,
                  x_axis_label=column,
                  y_axis_label="Count")
    plot.quad(bottom=0, top=column, left="left",
              right="right", source=src, fill_color=colors[0],
              line_color="#35838d", fill_alpha=0.7,
              hover_fill_alpha=0.7, hover_fill_color=colors[1])

    hover = HoverTool(tooltips=[('Interval', '@interval'),
                                ('Count', str("@" + column))])
    plot.add_tools(hover)

    output_notebook()
    show(plot)


# compute the number of bounding boxes per train image
# print(all_train_images.iloc[0])
all_train_images['count'] = all_train_images.apply(lambda row: 1 if any(row.bbox) else 0, axis=1)
train_images_count = all_train_images.groupby('name').sum().reset_index()
# print(train_images_count)
hist_hover(train_images_count, 'count', title='每张图中的瑕疵数量')











exit()