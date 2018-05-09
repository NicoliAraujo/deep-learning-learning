if __name__ == '__main__':
    import pandas as pd

    import os

    df_imdb = pd.read_csv('../../ssd_keras/dataset/csv/imdb_csv/imdb.csv', index_col=0)

    from data_generator.batch_generator import BatchGenerator

    train_dataset = BatchGenerator(box_output_format=['class_id'], 
                                   task='classification')

    train_dataset.parse_csv(labels_filename='../../ssd_keras/dataset/csv/imdb_csv/imdb.csv',
                            images_dir='../../ssd_keras/dataset/',
                            input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'])

    img_height, img_width = (64, 64)

    train_generator = train_dataset.generate(batch_size=4,
                                             shuffle=True,
                                             ssd_train=False,
                                             returns={'processed_labels',
                                                      'filenames'},
                                             ssd_box_encoder=None,
                                             convert_to_3_channels=True,
                                             equalize=False,
                                             brightness=(0.5, 2, 0.5),
                                             flip=0.5,
                                             translate=False,
                                             scale=False,
                                             max_crop_and_resize=False,
                                             # This one is important because the Pascal VOC images vary in size
                                             random_pad_and_resize=False,
                                             # This one is important because the Pascal VOC images vary in size
                                             random_crop=False,
                                             crop=False,
                                             resize=(img_height, img_width),
                                             gray=False,
                                             limit_boxes=True,
                                             # While the anchor boxes are not being clipped, the ground truth boxes should be
                                             include_thresh=0.4)
 
    print(next(train_generator))