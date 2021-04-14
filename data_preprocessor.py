# Code for seperating the data from the dataset from the Kaggle 'RSNA Intracranial Haemorrhage Detection'
# into directories according to the first diagnosis found in the CSV file (as many patients have multiple diagnoses).
# This was done for convincence as the labels were not used for this task.
# The images produced are also a combination of the associated masks into a RBG image, with the monochromatic scan data in the red channel
# and the monochromatic mask data in the green channel. This allows for Keras's default data augmentation methods
# to be used on the image like they would be in any other image processing training process.

import glob
import numpy as np
from matplotlib import pyplot as plt
import os
from PIL import Image
import re
import numpy as np
import re

path_directory = 'path/to/data/directory/with/numbered/patients/in'
image_load_directory = path_directory + '/%s/brain/*.jpg'
save_directory = 'path/to/save/directory'
path_to_csv = 'path/to/hemorrhage_diagnosis.csv'

tags = ['intraventricular', 'intraparenchymal', 'subarachnoid', 'epidural', 'subdural', 'none']
frax_tags = ['no', 'yes']

# index 0 = patient, index 1 = slide number, index 2-7 = class, index 8 = fracture

haem_tags = np.genfromtxt(path_to_csv, delimiter=',')

blank_image = Image.new('L', (650, 650), 'black')
print(haem_tags.shape)

i = 49
image_number = 0
img_cache = np.zeros((650,650,1))

regex = re.compile(r'\d+')

# Step through each patient in the directory
for i in range(131):

    if i < 100:
        path = image_load_directory % ('0' + str(i))
        images = glob.glob(path)
    else:
        path = image_load_directory % (str(i))
        images = glob.glob(path)

    images.sort(reverse=True)

    mask_found = False

    # Step through each image in the brain window of each patient:
    for image in images:
        with open(image, 'rb') as file:

            if mask_found == True:

                img = Image.open(file).convert('L')
                img = np.array(img)
                img = np.expand_dims(img, axis=2)

                # As the mask is always found before the associated scan, this will combine the current
                # with the mask that has been cached in the iteration before.
                output = np.concatenate((img/255, img_cache/255, np.zeros((650, 650, 1))), axis=2)
                image = Image.fromarray(np.uint8(output*255))

                # Check the associated CSV file for patient information.

                # Extract slide index and patient index from the filepath
                # search the CSV file for a row which has both patient index in column 0 and slide index in column 1.
                indices = [int(s) for s in regex.findall(file.name)]
                index_2 = np.where(haem_tags[:, 0] == indices[0])
                
                for i in index_2[0]:

                    if haem_tags[i][1] == indices[1]:

                        # Find out which category it is in.
                        index3 = np.where(haem_tags[i, 1:8] == 1)

                        # Save the image in the diagonsis of the first directory 
                        # with associated fracture tag
                        filename ='image_%s_fracture:%s' % (image_number, frax_tags[int(haem_tags[i][8])])
                        save_path = save_directory + '/' + tags[index3[0][0]-1] + '/' + filename + '.png'
                        print(save_path)
                        image.save(save_path)

                mask_found = False

            else:
                
                # If the image is a mask (as all masks are labelled with 'HGE_SEG'):
                if str(file.name).find('HGE_Seg') > 0:
            
                    # Cache the image for the next iteration, where it will be saved with
                    # its respective mask.
                    img = Image.open(file).convert('L')
                    img_cache = np.array(img)
                    img_cache = np.expand_dims(img_cache, axis=2)
                    mask_found = True

                else:
                    
                    # This code will be executed if the scan has no associated mask
                    img = Image.open(file).convert('L')
                    img = np.array(img)
                    img = np.expand_dims(img, axis=2)

                    # As the image is non-pathological, save a blank mask in the green channel
                    output = np.concatenate((img/255, np.zeros((650, 650, 1)), np.zeros((650, 650, 1))), axis=2)
                    image = Image.fromarray(np.uint8(output*255))

                    # Extract slide index and patient index from the filepath
                    # search the CSV file for a row which has both patient index in column 0 and slide index in column 1
                    indices = [int(s) for s in regex.findall(file.name)]
                    index_2 = np.where(haem_tags[:, 0] == indices[0])
                    
                    for i in index_2[0]:

                        if haem_tags[i][1] == indices[1]:

                            index3 = np.where(haem_tags[i, 1:8] == 1)

                            # Save the image in the diagonsis of the first directory 
                            # with associated fracture tag
                            filename ='image_%s_fracture:%s' % (image_number, frax_tags[int(haem_tags[i][8])])
                            save_path = save_directory + '/' + tags[index3[0][0]-1] + '/' + filename + '.png'
                            print(save_path)
                            image.save(save_path)

                image_number += 1
                  
    i += 1