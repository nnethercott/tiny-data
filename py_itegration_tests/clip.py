# Load model directly
import open_clip
import torch
from PIL import Image

cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.to(device).eval()

tokenizer = open_clip.get_tokenizer('ViT-B-32')


image = preprocess(Image.open("./images/bats/0.jpeg")).unsqueeze(0)
text = tokenizer(["a bat", "a dog", "a cat"])

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image).to(device)
    text_features = model.encode_text(text).to(device)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    text_probs = torch.round(text_probs.detach().cpu(), decimals = 3)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
