import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image


def plotImagesByPaths (imagesNames, basePath, offset, nrows, ncols):
    lenght = len(imagesNames)
    cycles = lenght / (nrows * ncols)
    firstCont = 0
    while firstCont < cycles:
        figure, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, nrows * ncols), constrained_layout=True)
        cont = 0
        for i in imagesNames[offset * nrows * ncols: (offset + 1) * nrows * ncols]:
            im = img.imread(basePath + i)
            ax.ravel()[cont].imshow(im)
            ax.ravel()[cont].set_title(i)
            ax.ravel()[cont].set_axis_off()
            cont = cont + 1
        plt.show()
        firstCont = firstCont + 1
        offset = offset + 1


# function to print original and cropped image
def printOaCImage (originalPath, croppedPath, i):
    figure, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 10), constrained_layout=True)
    im = img.imread(originalPath + i)
    ax.ravel()[0].imshow(im)
    ax.ravel()[0].set_title(i)
    ax.ravel()[0].set_axis_off()
    im = img.imread(croppedPath + i)
    ax.ravel()[1].imshow(im)
    ax.ravel()[1].set_title('Cropped ' + i)
    ax.ravel()[1].set_axis_off()


def cropAllImages (imagesList, basePath, savePath='', new_width=500, new_height=500, show=True, save=True):
    if (savePath == ''):
        savePath = basePath

    for image in imagesList:
        cropImage(image, basePath, savePath=savePath, new_width=new_width, new_height=new_height, show=show, save=save)


def cropImage (image, basePath, savePath, new_width=500, new_height=500, show=True, save=True):
    if (savePath == ''):
        savePath = basePath
    im = Image.open(basePath + image)
    width, height = im.size  # Get dimensions
    if width < new_width:
        new_width = width
    if height < new_height:
        new_height = height

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    if (show):
        im.show()
    if (save):
        im.save(savePath + image, "JPEG")
