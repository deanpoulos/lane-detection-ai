import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import json
import os
import csv



def get_xz_from_uv(bbox):
    # ref : http://ksimek.github.io/2013/08/13/intrinsic/
    car_width = 2.0  # assume car width is always 2.0m...
    fx = 714.1526
    fy = 710.3725
    cx = 713.85
    cy = 327

    # distance calc ref: http://emaraic.com/blog/distance-measurement
    z_distance = fx * (car_width / (bbox['right'] - bbox['left']))

    # find the approximate centre of the car ref:
    bb_c = (bbox['right'] + bbox['left'])/2

    # calculate the approx world x location from the pixel u
    # ref slide 10 of: http://www.cse.psu.edu/~rtc12/CSE486/lecture12.pdf
    approx_x = (bb_c - cx )*z_distance/fx

    approx_x+=0.053439859801385325*z_distance-0.5385462033846319 #testing least-square adjustment, y distance
    z_distance+=0.6055511973113685*approx_x + 2.2209803015779133 #testing least-square adjustment, x distance
    approx_x+=0.0026834076878362386*(bbox['right'] + bbox['left'])/2 + -1.899805878278793

    return approx_x,z_distance


test_clip = 1000
debug = True



predicted_points = []
provided_points = []

for test_clip in range(1,test_clip+1):
    if debug:
        original = cv2.imread("data/benchmark_velocity_train/clips/" + str(test_clip) + "/imgs/040.jpg")
        img = original.copy()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        # Get the current reference
        ax = plt.gca()

    # Find bounding boxes and plot them
    with open("data/benchmark_velocity_train/clips/" + str(test_clip) + "/annotation.json") as json_file:
        print("Opened "+"data/benchmark_velocity_train/clips/" + str(test_clip) )
        data = json.load(json_file)
        for car_an in data:
            width = car_an['bbox']['right'] - car_an['bbox']['left']
            height = car_an['bbox']['top'] - car_an['bbox']['bottom']

            x, z = get_xz_from_uv(car_an['bbox'])

            predicted_points.append((z,x))
            provided_points.append((car_an['position'][0], car_an['position'][1]))

            if debug:
                # Create a Rectangle patch
                text_annotated = "p=("+str("%.1f"%car_an['position'][0])+","+str("%.1f"%car_an['position'][1])+")"
                text_annotated2 = "p_calc=("+str("%.1f"%z)+","+str("%.1f"%x)+")"

                rect = Rectangle((car_an['bbox']['left'], car_an['bbox']['bottom']), width, height, linewidth=1, edgecolor='g', facecolor='none')
                ax.annotate(text_annotated+"\n"+text_annotated2, xy=(car_an['bbox']['left'], car_an['bbox']['top']), xycoords='data',
                            xytext=(car_an['bbox']['left'], car_an['bbox']['top']-40), textcoords='data',
                            horizontalalignment='right', verticalalignment='top',color='red',fontsize=14,rotation=0
                            )
                # Add the patch to the Axes
                ax.add_patch(rect)
    if debug:
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()


print(len(predicted_points))

# import csv
# with open('position_data.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["sample_x", "predict_x","abs_error_x","percent_error_x","sample_y","predict_y","abs_error_y","percent_error_y"])
#     for i in range(len(predicted_points)):
#         print("writing sample "+str(i))
#         abs_err_x = abs(provided_points[i][0]- predicted_points[i][0])
#         abs_err_y = abs(provided_points[i][1] - predicted_points[i][1])

#         per_err_x = abs(100*abs_err_x/provided_points[i][0])
#         per_err_y =  abs(100*abs_err_y/provided_points[i][1])

#         writer.writerow([provided_points[i][0], predicted_points[i][0],abs_err_x,per_err_x,provided_points[i][1], predicted_points[i][1],abs_err_y,per_err_y])

