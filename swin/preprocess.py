from transformers import AutoFeatureExtractor

model_name = 'microsoft/swin-base-patch4-windows7-224'
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

def transform(batch):
    inputs = feature_extractor([x.convert('RGB') for x in batch['image']], return_tensor='pt')
    inputs['label'] = batch['labal']
    return inputs

prepared_ds = ds.with_transform(transform)
