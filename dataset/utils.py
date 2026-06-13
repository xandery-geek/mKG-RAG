from torchvision import transforms


def resize_short_side(image, short_side=640):
    # Get the original dimensions
    width, height = image.size
    
    if width >= height > short_side:
        new_height = short_side
        new_width = int(width * new_height / height)
    elif height >= width > short_side:
        new_width = short_side
        new_height = int(height * new_width / width)
    else:
        new_width = width
        new_height = height

    resized_image = transforms.functional.resize(image, (new_height, new_width))
    return resized_image


def image_path_collate_fn(batch):
    images, image_paths = zip(*batch)
    return images, image_paths


def image_text_collate_fn(batch):
    images, texts, image_paths = zip(*batch)
    return images, texts, image_paths


def vqa_collate_fn(batch):
    images, questions, answers, image_paths = zip(*batch)
    return images, questions, answers, image_paths