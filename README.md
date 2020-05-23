# Introduction #

This repository contains a code that can be used to select the contours of objects in images.

# Packages #

* Pandas - for input need csv file of list images.

* Numpy - our contours to use as object numpy.ndarray.

* json - contours loaded, stored and save to file format as json.

* OpenCV - use to show image and for mouse callback

# Preparing #

Before start it is need to make csv file contain list images.

# Checkpoint of code #

* Read the first image and create window to show it.

* Loading json-file for every image  early selected contours or pass if not selected contours

* Drawing contours on image

* To use mouse callback for selecting new contours. If any contour not correct - there is possibility to delete it.

* In finish selected contours save to json-file. And choose the following image.
