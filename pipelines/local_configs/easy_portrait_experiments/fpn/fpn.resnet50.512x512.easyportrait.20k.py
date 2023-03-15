_base_ = [
    '../../__base__/models/fpn_resnet50.py',
    '../../__base__/datasets/easyportait_512x512.py',
    '../../__base__/default_runtime.py',
    '../../__base__/schedules/schedule_20k_adamw.py'
]

model = dict(decode_head=dict(num_classes=9))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

data = dict(samples_per_gpu=1, workers_per_gpu=1)