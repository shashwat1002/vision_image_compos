# %%
import os
import torch
os.environ["HF_HOME"] = "/scratch/shashwat.s/.cache"
os.environ["TRANSFORMERS_CACHE"] = "/scratch/shashwat.s/.cache"
from datasets import load_dataset

mscoco = load_dataset('HuggingFaceM4/COCO')

# %%

sample_train = mscoco['train']
sample_train

# %%
sample_train[0]['image']
len(sample_train)

# %%
# initialize model
from scripts.utils.model_init import init_subject_model

model_dict = init_subject_model(
    model_name='laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
    model_type='clip',
    device="cuda"
)


# %%
model_dict['model']
# %%
from tqdm import tqdm
def image_caption_embedding_pair(model, processor, tokenizer, dataset, num_samples=100):
    source, target = [], []
    with torch.no_grad():    
        for i in tqdm(range(num_samples)):
            image = dataset[i]['image'].convert('RGB')
            caption = dataset[i]['sentences']['raw']
            image_input = processor(images=image, return_tensors="pt", padding=True).to("cuda")
            caption_input = tokenizer(caption, return_tensors="pt", padding=True).to("cuda")
            embed_img = model.visual_projection(model.vision_model(**image_input).pooler_output).to("cpu")
            embed_txt = model.text_projection(model.text_model(**caption_input).pooler_output).to("cpu")
            # print(embed_img.shape, embed_txt.shape)
            source.append(embed_img.numpy())
            target.append(embed_txt.numpy())
    return source, target




# %%
source, target = image_caption_embedding_pair(model_dict['model'], model_dict['processor'], model_dict['tokenizer'], sample_train, 1000)

# %%

target.squeeze().shape
# %%
# use the linear ot library and make an ot

import ot
import numpy as np
source = np.stack(source).squeeze()
target = np.stack(target).squeeze()
ot_image_to_text = ot.da.LinearTransport(reg=1e-5)
ot_image_to_text.fit(Xs=source, Xt=target)

ot_text_to_image = ot.da.LinearTransport(reg=1e-5)
ot_text_to_image.fit(Xs=target, Xt=source)

# save both
import torch
ot_text_to_image_torch = torch.nn.Linear(1024, 1024, bias=True)
ot_text_to_image_torch.weight.data = torch.tensor(ot_text_to_image.A_)
ot_text_to_image_torch.bias.data = torch.tensor(ot_text_to_image.b_)

ot_image_to_text_torch = torch.nn.Linear(1024, 1024, bias=True)
ot_image_to_text_torch.weight.data = torch.tensor(ot_image_to_text.A_)
ot_image_to_text_torch.bias.data = torch.tensor(ot_image_to_text.b_)

torch.save(ot_text_to_image_torch, "ot_text_to_image.pt")
torch.save(ot_image_to_text_torch, "ot_image_to_text.pt")





# %%

source_train = source[:800, :]
target_train = target[:800, :]
source_test = source[800:, :]
target_test = target[800:, :]

train = np.stack([source_train, target_train], axis=1)
test = np.stack([source_test, target_test], axis=1)

train_y = np.array([0]*source_train.shape[0] + [1]*target_train.shape[0])
test_y = np.array([0]*source_test.shape[0] + [1]*target_test.shape[0])

# %%

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# import SGD
from sklearn.linear_model import SGDClassifier

# import MLP
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(max_iter=10000, hidden_layer_sizes=[500, 200, 100]).fit(train.reshape(-1, 1024), train_y)

# clf = LogisticRegression(random_state=0, max_iter=10000).fit(train.reshape(-1, 1024), train_y, )
y_pred = clf.predict(test.reshape(-1, 1024))

# %%

print(classification_report(test_y, y_pred))
