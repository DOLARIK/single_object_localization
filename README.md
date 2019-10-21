# Object Localization

This project was done as a part of a Hackathon hosted by Flipkart, by the name Flipkart GRID Challenge - _Teach the Machines_ (powered by Dare2Complete).

As the name suggests, the goal was to put a bounding box around an object in the image.

We were provided with:

1. 14000 _640x480x3_ coloured training images (each image had one object)
2. Image Labels corresponding to each image. (_labels were the __x,y__ coordinates of the __top-left__ and __bottom-right__ corners of the box_)

We were supposed to maximize the IOU (_intersection over union_) of our predicted bounding box. We were able to push that figure (_IOU_) upto __0.83__.

Competition was from _Feb, 2019_ to _March, 2019_. My friend [Divyank Pandey](https://github.com/pandeydivyank) was in it with me.

So I thought of uploading our work and maybe making it a little more reproducible.

## Results

<pre>  Round - 3:                                 Round - 2:</pre>

![Round-3](https://github.com/DOLARIK/single_object_localization/blob/master/submissions/submit_source_round_3/output_38_1.png) ![Round-2](https://github.com/DOLARIK/single_object_localization/blob/master/submissions/submit_source_round_2/output_40_0.png)

You can download:

1. Prepared [Dataset](https://drive.google.com/open?id=1RVoKzP6IeulTmuLRg6cgsoVEr3RcJAe-)
2. Pretrained models:
    - [Round - 3](https://drive.google.com/file/d/1WlQhOJZz4EEMkm8B6Y83trPUUEfya83C/view?usp=sharing)
    - [Round - 2](https://drive.google.com/open?id=1Ed81aWjZ_tH_CdjbC7j2RddalnuWYpmh)

Refer _Approach Text(s)_ for [Round - 2](https://github.com/DOLARIK/single_object_localization/blob/master/submissions/submit_source_round_2/Approach_Text.pdf) and [Round - 3](https://github.com/DOLARIK/single_object_localization/blob/master/submissions/submit_source_round_3/Approach_Text.pdf) for understanding the correct way of using the above.
