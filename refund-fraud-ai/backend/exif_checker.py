from PIL import Image, ExifTags
import datetime

def check_exif(image_path, delivery_time):
    img = Image.open(image_path)
    exif = img._getexif()

    if not exif:
        return False

    for tag, value in exif.items():
        if ExifTags.TAGS.get(tag) == "DateTimeOriginal":
            img_time = datetime.datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
            return abs((img_time - delivery_time).seconds) < 1800

    return False
