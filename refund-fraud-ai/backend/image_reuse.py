from PIL import Image
import imagehash
import os

def is_reused(image_path, db="backend/stored_images"):
    os.makedirs(db, exist_ok=True)
    new_hash = imagehash.phash(Image.open(image_path))

    for img in os.listdir(db):
        old_hash = imagehash.phash(Image.open(f"{db}/{img}"))
        if abs(new_hash - old_hash) < 8:
            return True

    Image.open(image_path).save(f"{db}/{os.path.basename(image_path)}")
    return False
