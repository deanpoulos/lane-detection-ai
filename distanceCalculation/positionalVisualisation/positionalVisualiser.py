import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.image as mpimg
import numpy as np

# Assumes straight roads/lanes
# carList = [(carX, carY), ...] Dimensions in Meters, position to rear-middle of vehicle. X, Y units in spec format.
# laneList = [laneOffsetY, ...] Dimensions in Meters, position of each lane line
def makeVisualisation(carList, laneList, label=""):
    # create figure, format
    fix, ax = plt.subplots(figsize=(5, 8))
    ax.set_xlim([-20, 20])
    ax.set_ylim([0, 120])
    ax.set_facecolor("#4f4f4f")
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.grid(True)
    ax.set_xticks(np.linspace(-20, 20, 11))
    ax.set_ylabel("X Distance (m)")
    ax.set_xlabel("Y Distance (m)")
    if label != "":
        ax.set_title(label)

    # read image
    green = mpimg.imread("green.png")
    # create OffsetImage object
    imBoxGreen = OffsetImage(green, zoom=0.04)
    # Green vehicle positon (front middle)
    xyGreen= [0, 0]
    # create annotation box object, includes specific image position details
    abGreen = AnnotationBbox(imBoxGreen, xyGreen, xybox=(0, -12), boxcoords='offset points', frameon=False)
    # Add green car to image
    ax.add_artist(abGreen)

    red = mpimg.imread("red.png")
    imBoxRed = OffsetImage(red, zoom=0.048)
    for car in carList:
        xyRed= (car[1], car[0])
        abRed = AnnotationBbox(imBoxRed, xyRed, xybox=(0, 20), boxcoords='offset points', frameon=False)
        ax.add_artist(abRed)
    
    # Line from bottom of graph to top
    yPoints = np.array([0, 120])
    for lane in laneList:
        # graph x points, image y points. matching yPoints
        xPoints = np.array([lane, lane])
        plt.plot(xPoints, yPoints, linestyle = 'dashed', color='y')

    plt.show(block=False)


# EXAMPLE:
if __name__ == "__main__":
    exampleCars = [(2, -4.5), (10, -4.5),(20, 0), (15, 6.5), (70, 4.5)]
    exampleLanes = [-1.5, 1.5, -4.5, -7.5, -10.5, 4.5, 7.5, 10.5]
    makeVisualisation(exampleCars, exampleLanes, "example")
    
    results = [(119.02543333333332, -9.308226666666654),
    (102.0218, -10.692811428571433),
    (75.17421111313243, -15.826419625834577),
    (42.00897647058824, -13.932371764705882),
    (79.35028888888888, -2.9832977777777936),
    (29.149047642060253, -15.402001107590406),
    (25.50545, -6.351785714285715)]

    makeVisualisation(results, exampleLanes, "Clip 31")


    completedNums = []
    # exampleResults = [{'imNumber': 1,
    #     'imShape': (720, 1280, 3),
    #     'bbox': {'top': 340.3418273926,
    #     'right': 868.1537475586,
    #     'bottom': 393.3275146484,
    #     'left': 784.6638793945},
    #     'provided': (21.30970213, 2.7243980956),
    #     'predicted': (21.188775617109375, 3.072024315210449)},
    #     {'imNumber': 2,
    #     'imShape': (720, 1280, 3),
    #     'bbox': {'top': 333.6641235352,
    #     'right': 867.5598144531,
    #     'bottom': 377.7915344238,
    #     'left': 802.6995849609},
    #     'provided': (34.9200542691, 6.2870887236),
    #     'predicted': (26.893357757135668, 4.377992506888333)},
    #     {'imNumber': 3,
    #     'imShape': (720, 1280, 3),
    #     'bbox': {'top': 340.1323547363,
    #     'right': 651.55859375,
    #     'bottom': 368.0708007812,
    #     'left': 617.0083618164},
    #     'provided': (42.5482547368, -2.7139237674),
    #     'predicted': (41.78353202927858, -2.9351904560526814)},
    #     {'imNumber': 4,
    #     'imShape': (720, 1280, 3),
    #     'bbox': {'top': 344.2061767578,
    #     'right': 730.0484619141,
    #     'bottom': 385.2985839844,
    #     'left': 679.7781982422},
    #     'provided': (28.3564872865, -0.3475407654),
    #     'predicted': (31.011534424993368, 0.6242702516753549)},
    #     {'imNumber': 5,
    #     'imShape': (720, 1280, 3),
    #     'bbox': {'top': 335.985534668,
    #     'right': 863.7344360352,
    #     'bottom': 374.0659179688,
    #     'left': 807.0223388672},
    #     'provided': (33.5129580944, 6.1829413897),
    #     'predicted': (30.490336879212663, 5.093148276152334)},
    #     {'imNumber': 5,
    #     'imShape': (720, 1280, 3),
    #     'bbox': {'top': 340.0504455566,
    #     'right': 730.8725585938,
    #     'bottom': 359.7734069824,
    #     'left': 704.3046875},
    #     'provided': (66.5060049972, 2.444539057),
    #     'predicted': (57.56563248503313, 2.6158531626581674)},
    #     {'imNumber': 6,
    #     'imShape': (720, 1280, 3),
    #     'bbox': {'top': 342.2452087402,
    #     'right': 774.2470092773,
    #     'bottom': 363.0127868652,
    #     'left': 748.1045532227},
    #     'provided': (66.5203254094, 7.5633314535),
    #     'predicted': (60.49082413864913, 6.001772417960219)},
    #     {'imNumber': 7,
    #     'imShape': (720, 1280, 3),
    #     'bbox': {'top': 343.4109802246,
    #     'right': 726.2653198242,
    #     'bottom': 358.6686401367,
    #     'left': 708.6020507812},
    #     'provided': (78.4947260368, 4.6407823801),
    #     'predicted': (85.62037147005263, 4.188540274419951)},
    #     {'imNumber': 8,
    #     'imShape': (720, 1280, 3),
    #     'bbox': {'top': 344.369934082,
    #     'right': 702.4273071289,
    #     'bottom': 381.3611450195,
    #     'left': 655.3842773438},
    #     'provided': (27.5100711491, -0.3545257629),
    #     'predicted': (32.33943608459728, -0.4016499587399205)},
    #     {'imNumber': 9,
    #     'imShape': (720, 1280, 3),
    #     'bbox': {'top': 344.3013305664,
    #     'right': 847.6300048828,
    #     'bottom': 366.8212890625,
    #     'left': 810.8087768555},
    #     'provided': (49.8523355317, 9.8339677496),
    #     'predicted': (45.73506616843394, 7.800860922098089)},
    #     {'imNumber': 10,
    #     'imShape': (720, 1280, 3),
    #     'bbox': {'top': 347.4986877441,
    #     'right': 685.7454833984,
    #     'bottom': 375.3241271973,
    #     'left': 650.6055908203},
    #     'provided': (36.67238106, -0.731694659),
    #     'predicted': (42.28228073443029, -0.9659947536705134)}]
    # for i in exampleResults:
    #     if (i["imNumber"] in completedNums):
    #         continue

    #     completedNums.append(i["imNumber"])
    #     curr_results = []
    #     for j in results[:11]:
    #         if (j["imNumber"]==i["imNumber"]):
    #             curr_results.append(j["predicted"])

    #     makeVisualisation(curr_results, exampleLanes, f"Image number: {i['imNumber']}")        
        


    plt.show()