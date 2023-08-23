from datasets import load_dataset

ds = load_dataset('food101')

ex = ds['train'][400]
print(ex)

# image = ex['image']
# image.show()

labels = ds['train'].features['label']
print(labels)

print(labels.int2str(ex['label']))

