import os
import sys

import open_clip
import torch
from PIL import Image
from progress.bar import Bar

CUDA = torch.cuda.is_available()
DEVICE = "cuda" if CUDA else "cpu"

def load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.eval().to(DEVICE)

    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    return model, preprocess, tokenizer 

def compute_cosine(image_features, text_features):
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    sim = image_features @ text_features.T
    return sim

def remove_images(imgs):
    for i in imgs:
        os.remove(i)

def filter_directory(dir, threshold = 0.2):
    model, preprocess, tokenizer = load_clip()

    # images
    imgs = [os.path.join(dir,i) for i in os.listdir(dir)]
    # imgs = sorted(imgs, key = lambda x: int(x.split('/')[-1][:-5]))

    tensors = torch.cat([preprocess(Image.open(i)).unsqueeze(0) for i in imgs])
    all_image_features = model.encode_image(tensors).to(DEVICE)

    # text
    tokens = tokenizer([dir])
    text_features = model.encode_text(tokens).to(DEVICE)

    # cosine
    cosines = compute_cosine(all_image_features, text_features)
    cosines = cosines.detach().cpu().squeeze(1)

    # filter 
    ids = torch.flatten((cosines<threshold).nonzero()).tolist()
    bad_imgs = [f for i,f in enumerate(imgs) if i in ids]
    remove_images(bad_imgs)

    return len(ids)


# pretty 
def filter_directories(dirs, threshold=0.2):
    count = 0
    with Bar('Filtering ... ', max = len(dirs)) as bar:
        for d in dirs:
            count+=filter_directory(d, threshold)
            bar.next()

    sys.stdout.write("\033[F")
    sys.stdout.write("\033[K")
    print(f'🔎 Filtered out {count} images.')

