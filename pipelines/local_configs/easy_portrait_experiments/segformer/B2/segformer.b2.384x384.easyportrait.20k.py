_base_ = [
    '../../../__base__/models/segformer.py',
    '../../../__base__/datasets/easyportait_384x384.py',
    '../../../__base__/default_runtime.py',
    '../../../__base__/schedules/schedule_20k_adamw.py'
]

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'  # noqa

model = dict(
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
                  embed_dims=64,
                  num_layers=[3, 4, 6, 3]),
    decode_head=dict(num_classes=9, in_channels=[64, 128, 320, 512]))

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