

def center_crop_image(image, resolution):
    distance = resolution // 2
    center = image.shape[2] // 2
    return image[:, :, center-distance:center+distance, center-distance:center+distance]


