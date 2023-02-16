#!/usr/bin/env python3

from UNETGAN.models import create_models, build_graph
from tqdm import tqdm
import gdal
import numpy as np
import cv2
import os 
c,inte,dis = create_models(n_channels=3)
c.load_weights('weights//classifier_10.hdf')
imagesize = 256

def predict_sheet(classifier, input_sheet_location, output_sheet_location, batch_size = 1, resolution = 1.25, padding = 0, img_size=imagesize, upsample=False):
    if upsample:
        padding = int(padding / 2)
        img_size = int(img_size / 2)
    else:
        ds = gdal.Open(input_sheet_location)

    transform_in = ds.GetGeoTransform()
    if upsample:
        transform_out = (transform_in[0], transform_in[1] / 2, transform_in[2],
                         transform_in[3], transform_in[4], transform_in[5] / 2)
    else:
        transform_out = (transform_in[0], transform_in[1], transform_in[2], transform_in[3], transform_in[4], transform_in[5])


    bands = []
    shape = np.shape(ds.ReadAsArray())
    print('shape of ds', shape)
    
    for i in range(shape[0]):
        bands.append(ds.GetRasterBand(i+1).ReadAsArray())
    if shape[0] ==3:
        bands.append(255*np.ones([shape[1],shape[2]]))
    sheet = np.dstack(tuple(bands))

    # make sure that the sheet matches the model extent parameters
    excess_x = sheet.shape[1] % img_size
    excess_y = sheet.shape[0] % img_size
    
    if not excess_x == 0:
        additional_padding_x = img_size - excess_x
    else:
        additional_padding_x = 0
    
    if not excess_y == 0:
        additional_padding_y = img_size - excess_y
    else:
        additional_padding_y = 0


    if upsample:
        sheet_template = np.zeros((sheet.shape[0] * 2, sheet.shape[1] * 2, 4), np.float32)
        
    else:
        sheet_template = np.zeros((sheet.shape[0] + additional_padding_y, sheet.shape[1] + additional_padding_x, 4), np.float32)
        
        
    sheet_extended = np.zeros((int(sheet.shape[0] + 2*padding + additional_padding_y), int(sheet.shape[1] + 2*padding + additional_padding_x), 4), np.float32)
    sheet_extended[padding:-padding-additional_padding_y, padding:-padding-additional_padding_x,:] = sheet

    x_count = int((sheet_extended.shape[1] - padding * 2) / img_size)
    y_count = int((sheet_extended.shape[0] - padding * 2) / img_size)

    
    for y in range(y_count):
        print(str(y) + str("/") + str(y_count))
        for x in range(x_count):
            y_start = y * img_size
            y_end = y * img_size + img_size
            x_start = x * img_size
            x_end = x * img_size + img_size

            sub_img = sheet_extended[y_start:y_end + 2 * padding, x_start:x_end + 2 * padding] / 255
            if upsample:
                sub_img = cv2.resize(sub_img, (sub_img.shape[1] * 2, sub_img.shape[0] * 2), interpolation = cv2.INTER_LINEAR)

            sub_img_expanded = np.expand_dims(sub_img, axis=0)

            sub_img_expanded = sub_img_expanded[:,:,:,0:3]
            
            Y_pred = classifier.predict(sub_img_expanded, batch_size)[0]

            if upsample:
                y_start_upsample = y * img_size * 2
                y_end_upsample   = y * img_size * 2 + img_size * 2
                x_start_upsample = x * img_size * 2
                x_end_upsample   = x * img_size * 2 + img_size * 2
                sheet_template[y_start_upsample:y_end_upsample, x_start_upsample:x_end_upsample, :] = Y_pred.copy()
                
            else:
                sheet_template[y_start:y_end, x_start:x_end, :] = Y_pred.copy()
    
    sheet_out = sheet_template[0:sheet.shape[0], 0:sheet.shape[1], :]
    print('the size of sheet out', np.shape(sheet_out))

    # write raster
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(output_sheet_location, sheet_out.shape[1], sheet_out.shape[0], 4, 
                            gdal.GDT_Float32, ['COMPRESS=LZW'])
    outdata.SetGeoTransform(transform_out)

    outdata.SetProjection(ds.GetProjection())
    sheet_out[sheet_out<=0.05] = 0

    for i in range(4):
        outdata.GetRasterBand(i+1).WriteArray(np.squeeze(sheet_out[:, :, i]))

    outdata.FlushCache()
    outdata = None

    ds = None   

if os.path.isdir("prediction") == False:
    os.mkdir("prediction")

with open("Datasamples//testingsheets.txt","r") as file:  
    test_validate_sheets = file.readlines()

for sheet in tqdm(test_validate_sheets):
    file = sheet.split('.')[0]
    target_dir = "prediction//" 
    base_dir = "testingsheets//"
    file_raw = file + ".tif"
    input_sheet_location = base_dir + file_raw
    output_sheet_location = target_dir + file + "_predictions.tif"
    predict_sheet(c, input_sheet_location, output_sheet_location, batch_size = 1, resolution = 1.25, padding = 0, img_size=imagesize, upsample=False)
