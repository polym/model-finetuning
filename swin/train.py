import torch
import numpy as np
from datasets import load_dataset, load_metric
from transformers import Trainer, TrainingArguments, SwinForImageClassification, AutoFeatureExtractor

ds = load_dataset('food101')

model_name = 'microsoft/swin-base-patch4-window7-224'
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)


def transform(batch):
    inputs = feature_extractor([x.convert('RGB') for x in batch['image']], return_tensor='pt')
    inputs['label'] = batch['label']
    return inputs


prepared_ds = ds.with_transform(transform)


def collate_fn(batch):
#    print([x['pixel_values'] for x in batch])
    return {
        'pixel_values': torch.stack([torch.from_numpy(x['pixel_values']) for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }


metric = load_metric('accuracy')
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


labels = ds['train'].features['label'].names

model = SwinForImageClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}, 
    ignore_mismatched_sizes = True,
)


batch_size = 16

training_args = TrainingArguments(
    f'swin-finetuned-food101',
    remove_unused_columns=False,
    evaluation_strategy = 'epoch',
    save_strategy = 'epoch',
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args = training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_ds['train'],
    eval_dataset=prepared_ds['validation'],
    tokenizer=feature_extractor,
)

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics('train', train_results.metrics)
trainer.save_metrics('train', train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate(prepared_ds['validation'])
trainer.log_metrics('eval', metrics)
trainer.save_metrics('eval', metrics)
