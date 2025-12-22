from PIL import Image
import imagehash
import os

HASH_DB = "hash_db.txt"

def is_image_reused(image_path, threshold=5):
    img = Image.open(image_path)
    img_hash = imagehash.phash(img)

    if not os.path.exists(HASH_DB):
        with open(HASH_DB, "w") as f:
            f.write(str(img_hash) + "\n")
        return False, 0

    with open(HASH_DB, "r") as f:
        hashes = f.readlines()

    for h in hashes:
        diff = img_hash - imagehash.hex_to_hash(h.strip())
        if diff <= threshold:
            return True, diff

    with open(HASH_DB, "a") as f:
        f.write(str(img_hash) + "\n")

    return False, 0
