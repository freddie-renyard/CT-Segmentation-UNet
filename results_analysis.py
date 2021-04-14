import os
import numpy as np
import pandas as pd
import glob
from skimage.io import imread
import matplotlib.pyplot as plt
import matplotlib.font_manager as fonts
from PIL import Image
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, recall_score, precision_score
import scipy.stats
from tensorflow import keras
import random

# Setting pyplot font to Helvetica

font_dir = '/path/to/font/directory'
font_files = fonts.findSystemFonts(fontpaths=font_dir)

for font_file in font_files:
    fonts.fontManager.addfont(font_file)

plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 16

# Setup directories for loading the model to test and the testing data, along with
# the directory to save figures in

model_name = 'MODEL_NAME.h5'

dir_names = ['intraventricular', 'intraparenchymal', 'subarachnoid', 'epidural', 'subdural', 'none']
base_path = 'path/to/test/directory/%s/*.png'
save_path = 'path/to/save/directory/'

# Extract the images (which have scan data in their red channel and mask data in the blue channel)
# from the directories specified and append them into a NumPy array.

i = 0
j = 0

for j in range(6):
    
    path = base_path % (dir_names[j])
    images = glob.glob(path)

    for image in images:
            with open(image, 'rb') as file:
                img = Image.open(file)
                img = img.resize((256,256))
                img = np.array(img)

                if (i == 0):
                    scans = img[:, :, 0:1]
                    masks = img[:, :, 1:2]
                    labels = j
                
    
                scans = np.concatenate((scans, img[:, :, 0:1]), axis=2)
                masks = np.concatenate((masks, img[:, :, 1:2]), axis=2)
                labels = np.append(labels, j)

                i = i + 1
    j = j + 1

# Process the arrays so that they are in the correct dimention format for input into the
# model.
scans = np.rollaxis(scans, 2)
masks = np.rollaxis(masks, 2)

# Normalise the pixels in the scan from (0-255) to (0-1).
scans = scans/255
masks = masks/255

# Load model and generate predictions.
model = keras.models.load_model(('path/to/model/' + model_name))
predictions = model.predict_on_batch(scans)
predictions = predictions[:, :, :, 0]

# Calculate the ROC curve and AUC for the entire test set, pixelwise.

ground_truths = masks.ravel()

fpr, tpr, _ = roc_curve(ground_truths.astype(int), predictions.ravel())
roc_auc = auc(fpr, tpr)

fig, (ax4) = plt.subplots(1,1, figsize=(7,7))

ax4.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % roc_auc)
ax4.plot([0,1], [0,1], 'k--')
ax4.set_xlim([0.0, 1.0])
ax4.set_ylim([0.0, 1.05])
ax4.set_xlabel('False Positive Rate')
ax4.set_ylabel('True Positive Rate')
ax4.set_title('ROC curve for Pixel-wise Predictions')
ax4.legend(loc='lower right')

print('')
print('Saving Pixel-wise ROC curve to directory...')
path = save_path + 'Pixelwise_ROC'
plt.savefig((path), dpi=300)
print('Finished!')
plt.close()

# Calculate ROC Curve and AUC case-wise

for q in range(masks.shape[0]):
    
    if (q == 0):
        mask_means = np.mean(masks[q, :, : ])
        prediction_means = np.mean(predictions[q, :, :])
    else:
        mask_means = np.append(mask_means, np.mean(masks[q, :, : ]))
        prediction_means = np.append(prediction_means, np.mean(predictions[q, :, :]))

mask_means_continuous = mask_means
mask_means = (mask_means != 0)

fpr, tpr, _ = roc_curve(mask_means.astype(int), prediction_means)
roc_auc = auc(fpr, tpr)
fig, (ax4) = plt.subplots(1,1, figsize=(7,7))

ax4.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % roc_auc)
ax4.plot([0,1], [0,1], 'k--')
ax4.set_xlim([0.0, 1.0])
ax4.set_ylim([0.0, 1.05])
ax4.set_xlabel('False Positive Rate')
ax4.set_ylabel('True Positive Rate')
ax4.set_title('ROC curve for Case-wise Predictions')
ax4.legend(loc='lower right')

print('')
print('Saving Case-wise ROC curve to directory...')
path = save_path + 'Casewise_ROC'
plt.savefig((path), dpi=300)
print('Finished!')
plt.close()

# Analyse the ability for the model to predict haemorrhage volume

# A step resolution of sampling the threshold of 100 is used.

step_resolution = 100
p = 0
for p in range(step_resolution):
    q = 0
    for q in range(127):
        if (q == 0):
            temp = np.mean((predictions[q, :, : ].ravel() > p/step_resolution))
        else:
            temp = np.append(temp, np.mean((predictions[q, :, : ].ravel() > p/step_resolution)))
    
    if (p == 0):
        temp2 = scipy.stats.pearsonr(mask_means_continuous, temp)[0]
        temp3 = scipy.stats.pearsonr(mask_means_continuous, temp)[1]
        p_range = p
    else:
        temp2 = np.append(temp2, scipy.stats.pearsonr(mask_means_continuous, temp)[0])
        temp3 = np.append(temp3, scipy.stats.pearsonr(mask_means_continuous, temp)[1])
        p_range = np.append(p_range, p)
    
# Check for undefined R values, which are returned as NaN by the Pearson 
# correlation coefficient function
for h in range(temp2.size):
    if (np.isnan(temp2[h])):
        temp2[h] = 0

highest_index = np.where(temp2 == np.amax(temp2))
print('')
print('Highest threshold-wise correlation coefficient: %0.4f' % temp2[highest_index[0]])
print('Corresponding P-value: ', temp3[highest_index])

m = keras.metrics.MeanIoU(num_classes=2)
m.update_state(masks.ravel(), (predictions.ravel() > int(highest_index[0])/step_resolution))
print('Corresponding IoU value: ', m.result().numpy())

optimal_threshold = int(highest_index[0])/step_resolution

# Create Precision-recall curves Cases

precision, recall, case_thresholds = precision_recall_curve(mask_means.astype(int), prediction_means)
pr_curve_auc = auc(recall, precision)

fig, (ax4) = plt.subplots(1,1, figsize=(7,7))

ax4.plot(recall, precision, label='PR curve (area = %0.4f)' % pr_curve_auc)
ax4.plot([0,1], [0,0], 'k--')
ax4.set_xlim([-0.05, 1.05])
ax4.set_ylim([-0.2, 1.05])
ax4.set_xlabel('Recall')
ax4.set_ylabel('Precision')
ax4.set_title('Precision-Recall Curve for Haemorrhage Classification')
ax4.legend(loc='lower right')

print('')
print('Saving Case-wise PR curve to directory...')
path = save_path + 'Casewise_PR'
plt.savefig((path), dpi=300)
print('Finished!')
plt.close()

# Determine the optimal threshold for classification using f-scores

i = 0
for threshold in case_thresholds:
    
    threshold_predictions = (prediction_means > threshold)
    f_score = f1_score(mask_means, threshold_predictions)

    if (i == 0):
        f_scores = f_score
    else:
        f_scores = np.append(f_scores, f_score)
    
    i = i + 1

highest_index = np.where(f_scores == np.amax(f_scores))
print('Highest case-wise F1 score: %0.4f' % f_scores[highest_index])

threshold_predictions = (prediction_means > case_thresholds[highest_index])
threshold_precision = precision_score(mask_means, threshold_predictions)
threshold_recall = recall_score(mask_means, threshold_predictions)
print('Respective highest case-wise precision: %0.3f, highest case-wise recall: %0.3f' % (threshold_precision, threshold_recall))

# Create pixel-based precision-recall curve

ground_truths = masks.ravel()/255

precision, recall, case_thresholds = precision_recall_curve(ground_truths.astype(int), predictions.ravel())
pr_curve_auc = auc(recall, precision)

fig, (ax4) = plt.subplots(1,1, figsize=(7,7))

ax4.plot(recall, precision, label='PR curve (area = %0.4f)' % pr_curve_auc)
ax4.plot([0,1], [0,0], 'k--')
ax4.set_xlim([-0.05, 1.05])
ax4.set_ylim([-0.2, 1.05])
ax4.set_xlabel('Recall')
ax4.set_ylabel('Precision')
ax4.set_title('Precision-Recall Curve for Pixel-wise Segmentations')
ax4.legend(loc='lower right')

print('')
print('Saving Pixel-wise PR curve to directory...')
path = save_path + 'Pixelwise_PR'
plt.savefig((path), dpi=300)
print('Finished!')
plt.close()

# Save plots of predictions in groups of 4

print('')
number_to_save = int(input("How many plots of 4 predictions would you like to save? "))

for t in range(number_to_save):
    index = 1
    plt.figure(figsize=(8,10))

    for z in range(4):

        prediction_index = random.randint(0, masks.shape[0])

        ax1 = plt.subplot(4, 3, index)
        ax1.imshow(scans[prediction_index, :, :], cmap='gray')
        ax1.axes.get_xaxis().set_visible(False)
        ax1.axes.get_yaxis().set_visible(False)

        index = index + 1
        
        ax2 = plt.subplot(4, 3, index)
        ax2.imshow(scans[prediction_index, :, :], cmap='gray')
        ax2.imshow((predictions[prediction_index, :, :] > int(highest_index[0])/100), cmap='jet', alpha=0.5, vmin=0, vmax=1)
        ax2.axes.get_xaxis().set_visible(False)
        ax2.axes.get_yaxis().set_visible(False)

        index = index + 1

        ax3 = plt.subplot(4, 3, index)
        ax3.imshow(scans[prediction_index, :, :], cmap='gray')
        ax3.imshow(masks[prediction_index, :, :], cmap='jet', alpha=0.5)
        ax3.axes.get_xaxis().set_visible(False)
        ax3.axes.get_yaxis().set_visible(False)

        index = index + 1

        if (z == 0):
            ax1.set_title('Scan')
            ax2.set_title('Prediction')
            ax3.set_title('Mask')

        masks = np.delete(masks, prediction_index, axis=0)
        scans = np.delete(scans, prediction_index, axis=0)
        predictions = np.delete(predictions, prediction_index, axis=0)

    print('')
    print('Saving prediction set %s to directory...' % (t + 1))
    path = save_path + ('Predictions_%s' % (t + 1))
    plt.savefig((path), dpi=400)
    print('Finished!')
    plt.close()