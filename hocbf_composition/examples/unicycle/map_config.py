from attrdict import AttrDict as AD

# Make map_ configuration
map_config = dict(
    geoms=(
        ('box', AD(center=[2.0, 1.5], size=[2.0, 2.0], rotation=0.0)),
        ('box', AD(center=[-2.5, 2.5], size=[1.25, 1.25], rotation=0.0)),
        ('box', AD(center=[-5.0, -5.0], size=[1.875, 1.875], rotation=0.0)),
        ('box', AD(center=[5.0, -6.0], size=[3.0, 3.0], rotation=0.0)),
        ('box', AD(center=[-7.0, 5.0], size=[2.0, 2.0], rotation=0.0)),
        ('box', AD(center=[6.0, 7.0], size=[2.0, 2.0], rotation=0.0)),
        ('boundary', AD(center=[0.0, 0.0], size=[10.0, 10.0], rotation=0.0)),
    ),
    velocity=(2, [-1.0, 9.0]),

)
