import configparser
import os, math, sys
import PIL, PIL.ImageDraw
from pathlib import Path
import math
import dlib
import numpy as np
import cv2

from scipy import ndimage
from scipy.misc import imread, imsave, imresize

#from skimage.filters import threshold_adaptive #will be deprecated
from skimage.filters import threshold_local
from skimage.color import rgb2gray
from skimage import measure
#import matplotlib.pyplot as plt

from . import stomata_model

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 200, 'Batch size Must divide evenly into the dataset sizes.')


def image_whitening(img):
    img = img.astype(np.float32)
    d, w, h = img.shape
    num_pixels = d * w * h
    mean = img.mean()
    variance = np.mean(np.square(img)) - np.square(mean)
    stddev = np.sqrt(variance)
    min_stddev = 1.0 / np.sqrt(num_pixels)
    scale = stddev if stddev > min_stddev else min_stddev
    img -= mean
    img /= scale
    return img
def create_and_return_subdirectory(image_path):
    global name, sub_directory, v_directory
    '''create subdirectory and verbose directory with image name if not present.
    will return directory and subdirectory name as string.
    '''
    directory = os.path.dirname(image_path)  # need to add / at the end for other usage
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    folder_name = os.path.basename(directory)
    # create subdirectory

    sub_directory = os.path.join(directory, "annotated")
    if not os.path.exists(sub_directory):
        os.mkdir(sub_directory)

    v_directory = os.path.join(directory, "verbose")
    if not os.path.exists(v_directory):
        os.mkdir(v_directory)

    return name, folder_name, directory, sub_directory, v_directory
def import_config(config_path):
    '''
    check path of config file
    '''
    global ini

    print ("\tconfig file path:", config_path)
    c_file = Path(config_path)
    if c_file.is_file():
        print ("\tReading config file.")
    else:
        print ("\tCould not locate config file.")
        sys.exit()
    ini = configparser.ConfigParser()
    ini.read(config_path)

    if os.path.isabs(ini['path']['detector_path']) is False:
        ini['path']['detector_path'] = os.path.join(os.path.dirname(__file__), ini['path']['detector_path'])
    if os.path.isabs(ini['path']['classifier_path']) is False:
        ini['path']['classifier_path'] = os.path.join(os.path.dirname(__file__), ini['path']['classifier_path'])


    print ("\tValidating detector path")

    if Path(ini['path']['detector_path']).is_file():
        print ("\tDetector located")
    else:
        print ("\tCould not locate detector file at", ini['path']['detector_path'])
        sys.exit()

    print ("\tValidating classifier path")
    if Path(ini['path']['classifier_path']).is_file():
        print ("\tClassifier located")
    else:
        print ("\tCould not locate classifier file at", ini['path']['classifier_path'])
        sys.exit()

    #how to use.... ini.get('settings', 'host')
def check_type_of_input(dir_path):
    '''
    input: dir or file absolute path
    output: item file path or list of file path
    '''
    item_list = []
    if os.path.isfile(dir_path):
        #check format
        if dir_path.endswith(".jpg") or dir_path.endswith(".tiff") or dir_path.endswith(".tif") or dir_path.endswith(".jpeg"):
            item_list.append(dir_path)
            return item_list
        else:
            print ("input must be jpg, jpeg, tiff, or tif format.")
            sys.exit()
    elif os.path.isdir(dir_path):
        print ("directory detected. reading files inside directory.")
        for item in os.listdir(dir_path):
            if item.endswith(".jpg") or item.endswith(".tiff") or item.endswith(".tif") or item.endswith(".jpeg"):
                item_list.append(os.path.join(dir_path, item))
        if len(item_list) == 0:
            print ("no image files inside specified directory.")
            sys.exit()
    else:
        print ("not a valid file or directory")
        sys.exit()
    return item_list  # list of files
def stomata_detector(image_path, detector_path, detection_image_width=False):
    #will remove scaled image output in future.
    '''
    detects the position of the stomata

    input
        image_path: absolute path of the image file to be analyzed.
        detector: absolute path of the HOG detector constructed by dlib simple object detector
        detection_image_width: the image width the image file to be downscaled for analyses.
    output:
        image: numpy image by scipy.
        scaled_image : scaled image by scipy.
        original_dets* :coordinates of detected rectangle in original size.
        scaled_dets* : coordinates of detected rectangle in scaled_size.
        ratio : ratio of original/x_scaledwidth

        *detected coordinates list of stomata is realigned from left to right for convinience, and is not the "dets" format dlib regularly returns.

    if detection_image_width is not inputted,
    output:
        image: same as above.
        dets: coordinate lists of detected rectangle in scaled_size.


    '''

    # detector_extension check
    temp, detector_ext = os.path.splitext(detector_path)
    if (detector_ext != '.svm'):
        raise ValueError("extention of the detector must be .svm constructed by dlib hog.")
    # HOG+SVM detector by dlib
    detector = dlib.simple_object_detector(detector_path)

    image = imread(image_path)
    height = image.shape[0]
    width = image.shape[1]

    if detection_image_width:
        #print ("detection mode in converted size mode")
        # calculate scale that will resize the input image to the desired scale (x=512px in default) near detection width for stomata detection
        ratio = width / detection_image_width
        height_det = int(round(height / ratio))
        width_det = int(round(width / ratio))

        # prepare small size image for detection
        scaled_image = imresize(image, (height_det, width_det))

        # put detected position into array
        dets = detector(scaled_image)
        scaled_dets = []
        original_dets = []

        #due to the ratio calculation, detector converted to original scale sometimes exceeds the size of its image.
        #in case of that, it will be modified to the height and width of the image, respectively
        for d in dets:
            scaled_dets.append([d.left(), d.top(), d.right(), d.bottom()])

            templeft = d.left() * ratio
            tempright = d.right() * ratio
            if d.left() <= 0:
                templeft = 1
            if d.right() * ratio > width:
                tempright = width

            tempbottom = d.bottom() * ratio
            temptop = d.top() * ratio
            if d.bottom() * ratio > height:
                tempbottom = height
            if d.top() < 0:
                temptop = 1
            original_dets.append([round(templeft), round(temptop), round(tempright), round(tempbottom)])

        print ("\t",str(len(original_dets)) + " stomata detected")
        # reorder the detected region number from left to right
        scaled_dets = sorted(scaled_dets, key=lambda i: i[0], reverse=False)
        original_dets = sorted(original_dets, key=lambda i: i[0], reverse=False)

        return image, scaled_image, original_dets, scaled_dets, ratio

    else:
        dets = detector(image)
        dets2 = []
        #due to the ratio calculation, detector converted to original scale sometimes exceeds the size of its image.
        #in case of that, it will be modified to the height and width of the image, respectively
        for d in dets:

            templeft = d.left()
            tempright = d.right()
            if d.left() <= 0:
                templeft = 1
            if d.right() > width:
                tempright = width

            tempbottom = d.bottom()
            temptop = d.top()
            if d.bottom() > height:
                tempbottom = height
            if d.top() < 0:
                temptop = 1
            dets2.append([round(templeft), round(temptop), round(tempright), round(tempbottom)])
        #print (str(len(dets2)) + " candidate region detected for " + base_name)

        # reorder the detected region number from left to right
        dets = sorted(dets2, key=lambda i: i[0], reverse=False)
        return image, dets
def draw_stomata_position(image, coords, text=True):
    '''
    draws a rectangle outline to the image according to the input coordinate

    input
        coords: rectangle array [left,top,right,bottom]

    output
        image
    '''
    image_position = image.copy()
    if image_position.shape[0] > 1000:
        font_size = 2
        font_width = 8
        line_width = 3
    else:
        font_size = 0.5
        font_width = 1
        line_width = 1

    i = 1
    for d in coords:
        left = int(d[0])
        top = int(d[1])
        right = int(d[2])
        bottom = int(d[3])

        # draw rectangle to the detected region if text ==True.

        cv2.rectangle(image_position, (left, top), (right, bottom), (0, 0, 0), line_width)
        # used for adjusting the text overlaying position so that annotation will not be drawn outside the image.
        fixed_top = top
        if top < image_position.shape[0] * 0.4:
            fixed_top = bottom
        if text is True:
            # add region no.
            cv2.putText(image_position, str(i), (left, fixed_top), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), font_width, cv2.LINE_AA)
        i += 1

    return image_position
def crop_and_save_stomata_images(image, coords, name, v_directory):
    i = 1
    for d in coords:
        stomata = image[d[1]:d[3], d[0]:d[2]]
        file_name = os.path.join(v_directory, name + str("{0:02d}".format(i)) + ".jpg")
        imsave(file_name, stomata)
        i += 1
def create_montage (image, coords, column_size = 5):
    '''
    input
        image
        coordinate of the stomata position within a image obtained by stomata detector
    output

    '''
    i = 1
    maxwidth = 0
    maxheight = 0

    for d in coords:
        stomata = image[d[1]:d[3], d[0]:d[2]]
        if maxheight < stomata.shape[0]:
            maxheight = stomata.shape[0]
        if maxwidth < stomata.shape[1]:
            maxwidth = stomata.shape[1]
        i += 1
    margin = 1

    height = maxheight * math.ceil(i / column_size)
    width = (maxwidth + margin) * column_size
    montage = PIL.Image.new(mode='RGBA', size=(width, height), color=(0, 0, 0, 0))

    draw = PIL.ImageDraw.Draw(montage)

    n = 0
    for d in coords:
        stomata = image[d[1]:d[3], d[0]:d[2]]
        pilimg = PIL.Image.fromarray(np.uint8(stomata))

        xpos1 = (maxwidth + margin) * n
        #12345 -> -0 #678910 -xpos1*1
        subtractx = ((maxwidth + margin) * (column_size)) * math.floor(n / (column_size))
        xpos = xpos1 - subtractx
        ypos = (maxheight + margin) * math.floor(n / (column_size))

        montage.paste(pilimg, (xpos, ypos))
        text = str(n + 1)
        draw.text((xpos, ypos), text, (0, 0, 0))
        n += 1

    return montage
def pore_evaluator(image, no, offset=(0, 0), o_ver=False):
    def gaussian(img, sigma=1):
        return ndimage.gaussian_filter(img, sigma)
    def open_verbose(no):
        print ("running verbose mode")
        # verbose mode for outputting all the detected region into files per region.
        # used for debugging
        csv_verbose = open(v_directory + "/" + name + "_openverbose.csv", 'a+')
        csv_verbose.write(
            "Image_name,Stomata No.,LabelNo.,Aperture,major_axis_length,Area,solidity,centroid[0],centroid[1],\n")
        k_number = 0
        k = 0
        for k in regionprops:
            s = name + "," + str(no) + "," + str(k_number) + "," + str(k.minor_axis_length) + "," + str(
                k.major_axis_length) + "," \
                + str(k.area) + "," + str(k.solidity) + "," + str(k.centroid[0]) + "," + str(k.centroid[1]) + "\n"
            csv_verbose.write(s)
            k_number += 1
        csv_verbose.close()

        # print ("knumber is " + str(k_number)+"no.label is "+ str(nb_labels))

        # determine the number of rows required for image generation
        j = math.ceil((k_number / 4) + 1)
        fig, ax = plt.subplots(j, 4, figsize=(100, 100), dpi=100)
        # fig.tight_layout()
        ax = ax.ravel()
        for a in ax:
            a.axis('off')
        ax[0].imshow(image)
        ax[0].set_title("input")
        ax[1].imshow(image)
        ax[1].set_title("binary_image")
        ax[2].imshow(label_im, cmap="spectral")
        ax[2].set_title("detected labels")
        ax[3].imshow(im_filt, cmap="gray")
        ax[3].set_title("filter passed contour")
        # variable
        n = 0
        while n <= k_number - 1:
            # ax[n].imshow#### create one mask per region.!!!!!
            empty = np.zeros((height, width))
            empty[label_im == regionprops[n].label] = True
            ax[n + 4].axis('off')
            ax[n + 4].imshow(empty, cmap=plt.cm.gray)
            ax[n + 4].set_title(str(n))
            n += 1
        # plt.show()
        plot_directory = str(v_directory + name + "_" + str(no) + "_open_verbose.jpg")
        # print (plot_directory)
        fig.savefig(plot_directory)
    def semiverbose():
        fig, ax = plt.subplots(2, 2)
        # fig.tight_layout()
        ax = ax.ravel()
        for a in ax:
            a.axis('off')
        ax[0].imshow(image)
        ax[1].imshow(label_im, cmap='spectral')
        ax[2].imshow(im_filt, cmap='spectral')
        #framename = str(name) + "_frame_" + str(no)
        plot_directory = str(v_directory + "/" + name + "_imfilt_" + str(no) + ".png")
        fig.savefig(plot_directory)
        plt.close()

    image_with_pore = image.copy()
    gray = rgb2gray(image)
    gray = ndimage.gaussian_filter(gray, 3)
    #image2 = threshold_adaptive(gray, 31, 'gaussian')
    local_thresh = threshold_local(gray, 31, method="gaussian")
    image2 = gray > local_thresh

    image2 = ndimage.morphology.binary_opening(image2)
    image2 = ndimage.morphology.binary_closing(image2)

    label_im, nb_labels = ndimage.label(image2)
    regionprops = measure.regionprops(label_im, intensity_image=gray)
    im_filt = label_im > 0

    height = image.shape[0]
    width = image.shape[1]
    open_regionprops = []
    n_o_regions = 0

    for prop in regionprops:
        # define criteria and mask away unwanted regions
        if prop.area < int(ini['open_criteria']['min_area']) or \
         prop.area > int(ini['open_criteria']['max_area']) or \
         prop.solidity < float(ini['open_criteria']['min_solidity']) or \
         prop.major_axis_length < int(ini['open_criteria']['min_major_axis_length']) or \
         prop.centroid[0] < height * float(ini['open_criteria']['margin']) or \
         prop.centroid[0] > height * (1 - float(ini['open_criteria']['margin'])) or \
         prop.centroid[1] < width * float(ini['open_criteria']['margin']) or\
         prop.centroid[1] > width * (1 - float(ini['open_criteria']['margin'])):
            im_filt[label_im == prop.label] = False
        else:
            # retain
            open_regionprops.append(prop)
            n_o_regions += 1

    im_filt = ndimage.binary_fill_holes(im_filt)

    label_im2, nb_labels = ndimage.label(im_filt)
    regionprops = measure.regionprops(label_im2)
    n_o_regions = 0
    for prop in regionprops:
        n_o_regions += 1

    if n_o_regions == 1:
        #print ("    quantified")
        new_coords = np.empty((0, 2), int)
        for coords in open_regionprops[0].coords:
            # xy to yx conversion for drawing in fillPoly function
            new_coords = np.append(new_coords, np.array([[coords[1], coords[0]]]), axis=0)

        cv2.fillPoly(image_with_pore, [new_coords], (0, 255, 0), offset=(0, 0))
        stat = "open"
        return int(1), new_coords, open_regionprops, stat

    elif n_o_regions >= 2:
        #print ("    " + str(n_o_regions) + "regions remained after pore filtering. returning the largerst area with flag")
        l = 0
        temp = []
        while l < n_o_regions - 1:
            if open_regionprops[l].area > open_regionprops[l + 1].area:
                temp = []
                temp.append(open_regionprops[l])
                l += 1
            else:
                temp = []
                temp.append(open_regionprops[l + 1])
                l += 1

        new_coords = np.empty((0, 2), int)

        for coords in temp[0].coords:
            # xy to yx conversion for drawing in fillPoly function
            new_coords = np.append(new_coords, np.array([[coords[1], coords[0]]]), axis=0)
        stat = "open"
        cv2.fillPoly(image_with_pore, [new_coords], (0, 255, 0), offset=(0, 0))
        #semiverbose()
        return int(1), new_coords, temp, stat
    else:
        #semiverbose()
        #print ("        no valid region detected (01)")
        return int(0), int(0), int(0), int(0)
 
def stomata_stat_batch_classify(image, region_number, ckpt_path):

    '''
    input
        image : image read by scipy. imread if by opencv, bgr to rgb must be performed
        ckpt_path : checkpoint absolute path
    output
        most likely stat of stomata, confidential level of most likely stat of stomata
    '''
    DST_INPUT_SIZE = 56
    NUM_CLASS = 4
    tf.reset_default_graph()

    image = tf.reshape(image, [-1, DST_INPUT_SIZE, DST_INPUT_SIZE, 3])
    logits = stomata_model.tf_inference(image, region_number, DST_INPUT_SIZE, NUM_CLASS)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    if ckpt_path:
        saver.restore(sess, ckpt_path)
    softmax = tf.nn.softmax(logits).eval()
    results = [[None for _ in range(2)] for _ in range(region_number)]
    q = 0
    for logit in softmax:
        logit = [round(n * 100, 1) for n in logit]
        logit = np.asarray(logit)
        result = [["open", logit[0]], ["closed", logit[1]], ["partially_open", logit[2]], ["false_positive", logit[3]]]
        result = sorted(result, key =lambda x: int(x[1]), reverse = True)
        results[q][0] = result[0][0]
        results[q][1] = result[0][1]
        q += 1
    #print ("\t",results)
    return results
def text_top_position(d1, d3, y):
    if d1 < y * 0.2:
        return d3
    else:
        return d1
def analyze(image_path):
    '''parse per image.
        1. create input image file name subdirectory, get directory and subdirectory path
        2. detect stomata position in input image
        3. create image files.
        4. evaluate and analyze pore. per detected region
    '''
    # 1. create input image file name subdirectory, get directory and subdirectory path

    name, folder_name, directory, sub_directory, v_directory = create_and_return_subdirectory(image_path)

    ##################################################################################

    #2.detect stomata position in input image#######################

    original_dets = []
    try:
        image, scaled_image, original_dets, scaled_dets, ratio\
            = stomata_detector(image_path, ini['path']['detector_path'], detection_image_width=512)
    except Exception as e:
        print (e)
        print("no stomata detected. skipping annotation")

    #################################################################


    #3. create image files.##########################################

    if len(original_dets) > 0:  # if stomata was detected
        overlay_name = os.path.join(sub_directory, name + "_stomata_position.jpg")
        #montage_name = os.path.join(sub_directory, name + "_stomata_tiled.jpg")
        image_with_position = draw_stomata_position(image, original_dets)
        #imsave(overlay_name, image_with_position)
        #save stomata montage images
        #montage = create_montage(image, original_dets)
        #imsave(montage_name, montage)
        #save respective stomata
        crop_and_save_stomata_images(image, original_dets, name, v_directory)

    ##################################################################


        #4. obtain stomata stat. in batch by tensorflow###################

        stomata_all = []
        for d in original_dets:  # analyze per stomata.
            stomata = image[d[1]:d[3], d[0]:d[2]]
            stomata = imresize(stomata, (56, 56))
            stomata = image_whitening(stomata)
            stomata_all.append(stomata.flatten().astype(np.float32))  # /255.0)
        stomata_all = np.asarray(stomata_all)
        results = stomata_stat_batch_classify(stomata_all, len(original_dets), ini['path']['classifier_path'])

        ##################################################################


        #5. count stomata status statistics and write to csv file.########
        open_count = 0
        close_count = 0
        popen_count = 0
        nolabel_count = 0

        for result in results:
            if result[0] == "open":
                open_count += 1
            elif result[0] == "closed":
                close_count += 1
            elif result[0] == "partially_open":
                popen_count += 1
            elif result[0] == "false_positive":
                nolabel_count += 1
        #print (open_count, close_count, popen_count, nolabel_count)

        csv_class_count_path = os.path.join(directory, folder_name + "_classification_count.csv")
        csv_class_count = open(csv_class_count_path, 'a+')
        #header
        if os.stat(csv_class_count_path).st_size == 0:  # write header if empty
            csv_class_count.write("Image_name,open,partially_open,closed,false_positive\n")
        #inside
        s = ",".join([name, str(open_count), str(popen_count), str(close_count), str(nolabel_count) + "\n"])
        #print (s)
        csv_class_count.write(s)
        csv_class_count.close()
        ###################################################################



        #prepare output csv files
        csv_path = os.path.join(directory, folder_name + "_all.csv")
        csv_all = open(csv_path, 'a+')
        if os.stat(csv_path).st_size == 0:  # write header if empty
            csv_all.write("Image_name,RegionNo.,Stat,Aperture(um),Area(um^2),Long_axis_length(um),Aperture/Long_axis_length(arbitrary),Centroid(X)(px),Centroid(Y)(px),Eccentricity(arbitrary),Solidity(arbitrary)\n")


        base_image = image.copy()
        image_all_annotated = image.copy()
        image_classified = image.copy()

        q = 0
        no = 1  # no. of all detected region


        for d in original_dets:  # analyze and record per stomata.
            stomata = image[d[1]:d[3], d[0]:d[2]]
            region_stat = results[q][0]
            percentage = results[q][1]
            fixed_top = text_top_position(d[1], d[3], image_all_annotated.shape[0])

            if region_stat == "closed":
                #colors = (0,0,255)
                colors = (135,206,235)

                s = ",".join([name, str(no), "closed", str(0), str(0) + "\n"])
                csv_all.write(s)

                #cv2.rectangle(base_image, (d[0], d[1]), (d[2], d[3]), colors, 2)
                cv2.rectangle(image_all_annotated, (d[0], d[1]), (d[2], d[3]), colors, 3)


            elif region_stat == "false_positive":
                colors = (100,100,100)
                

                #cv2.rectangle(base_image, (d[0], d[1]), (d[2], d[3]), colors, 2)
                cv2.rectangle(image_all_annotated, (d[0], d[1]), (d[2], d[3]), colors, 3)
                

            elif region_stat == "open" or region_stat == "partially_open":
                if region_stat == "open":
                    colors = (255,0,0)
                elif region_stat == "partially_open":
                    colors = (255,165,0)

                region_number = 0

                try: 
                    region_number, new_coords, regionprops, stat = pore_evaluator(stomata, no, o_ver=False)
                except Exception as e:
                    s = ",".join([name, str(no), region_stat + "_but_failed_to_detect_pore", str(0), str(0) + "\n"])
                    csv_all.write(s)
                    #cv2.rectangle(base_image, (d[0], d[1]), (d[2], d[3]), colors, 2)
                    cv2.rectangle(image_all_annotated, (d[0], d[1]), (d[2], d[3]), colors, 2)

                #initialize um
                um = "n.d."
                if region_number > 0:
                    #draw stomatal pore
                    cv2.fillPoly(image_all_annotated, [new_coords], colors, offset=(d[0], d[1]))
                    for n in regionprops:  # regionprops may contain multiple areas, but new_coords contain only one
                        s = ",".join([
                            name, str(no), region_stat,
                            str(n.minor_axis_length / float(ini['misc']['pixel_per_um'])),
                            str(n.area / float(ini['misc']['pixel_per_um']) ** 2),
                            str(n.major_axis_length),
                            str(n.minor_axis_length / n.major_axis_length),
                            str(n.centroid[1]),
                            str(n.centroid[0]),
                            str(n.eccentricity),
                            str(n.solidity) + "\n"
                        ])
                        csv_all.write(s)

                        um = n.minor_axis_length / float(ini['misc']['pixel_per_um'])
                        #minor axis length is already written to csv, but have indivisualy calculate for drawing
                        y0, x0 = n.centroid
                        orientation = n.orientation                        
                        #x1 y1 is for major axis length
                        #x1 = x0 + math.cos(orientation) * 0.5 * n.major_axis_length
                        #y1 = y0 - math.sin(orientation) * 0.5 * n.major_axis_length
                        #x2,y2 minor axis length end point, x3,y3 minor axis length start point
                        x2 = x0 - math.sin(orientation) * 0.5 * n.minor_axis_length
                        y2 = y0 - math.cos(orientation) * 0.5 * n.minor_axis_length
                        x3 = x0 + math.sin(orientation) * 0.5 * n.minor_axis_length
                        y3 = y0 + math.cos(orientation) * 0.5 * n.minor_axis_length
                        cv2.arrowedLine(base_image, (math.floor(d[0]+x3), math.floor(d[1]+y3)), (math.floor(d[0]+x2), math.floor(d[1]+y2)), (0, 0, 0), thickness = 2,tipLength=0.3)
                        cv2.arrowedLine(base_image, (math.floor(d[0]+x2), math.floor(d[1]+y2)), (math.floor(d[0]+x3), math.floor(d[1]+y3)), (0, 0, 0), thickness = 2,tipLength=0.3)
                        cv2.arrowedLine(image_all_annotated, (int(d[0]+x3), int(d[1]+y3)), (int(d[0]+x2), int(d[1]+y2)), (0, 0, 0), thickness = 2,tipLength=0.3)
                        cv2.arrowedLine(image_all_annotated, (int(d[0]+x2), int(d[1]+y2)), (int(d[0]+x3), int(d[1]+y3)), (0, 0, 0), thickness = 2,tipLength=0.3)

                    #cv2.rectangle(base_image, (d[0], d[1]), (d[2], d[3]), colors, 2)
                    cv2.rectangle(image_all_annotated, (d[0], d[1]), (d[2], d[3]), colors, 3)
                else:
                    cv2.rectangle(image_all_annotated, (d[0], d[1]), (d[2], d[3]), colors, 3)
                    
            else:
                #cv2.rectangle(base_image, (d[0], d[1]), (d[2], d[3]), colors, 2)
                cv2.rectangle(image_all_annotated, (d[0], d[1]), (d[2], d[3]), colors, 3)
                

            #write rstat and aperture.
            
            if region_stat == "closed":
                aperture =  "0um"

            elif region_stat == "open" or region_stat == "partially_open":
                try:
                    aperture =  '%02.2f' % um + "um"
                except:
                    um = "n.d."
                    aperture =  um + "um"                    
            else:
                um = "not measured"
                aperture = um
            try:
                print(region_stat,aperture,end=",")
            except:
                print(region_stat,aperture)
            notxt = "No." + str(no)

            #adjust background size for text
            if region_stat == "open" or region_stat == "closed":
                pad = 150
            elif region_stat == "partially_open" or region_stat == "false_positive":
                pad = 250

            #cv2.rectangle(base_image, (d[0], fixed_top-20), (d[0]+250, fixed_top+10),colors,-1)
            cv2.rectangle(image_all_annotated, (d[0], fixed_top-50), (d[0]+pad,fixed_top+25),colors,-1)

            cv2.putText(base_image, region_stat, (d[0], fixed_top-25), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image_all_annotated, region_stat, (d[0], fixed_top-25), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            cv2.putText(base_image, aperture, (d[0], fixed_top+10), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image_all_annotated, aperture, (d[0], fixed_top+10), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            #cv2.putText(image_classified, text, (d[0], fixed_top), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            
            no += 1
            q += 1
        alpha=0.5
        cv2.addWeighted(image_all_annotated,alpha,base_image,1-alpha,0, image_all_annotated)
        csv_all.close()
        image_path_annotated = sub_directory + "/" + name + "_all.jpg"
        image_path_classified = sub_directory + "/" + name + "_classified.jpg"
        imsave(image_path_annotated, image_all_annotated)
        #imsave(image_path_classified, image_classified)
        print("  ")
