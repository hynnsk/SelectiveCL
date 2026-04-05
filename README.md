# Selective Contrastive Learning for Weakly Supervised Affordance Grounding (ICCV 2025)
WonJun Moon*</sup>, Hyun Seok Seong*</sup>, Jae-Pil Heo</sup> (*: equal contribution)

[[Arxiv](https://arxiv.org/abs/2508.07877)]

## Abstract
> Facilitating an entity’s interaction with objects requires accurately identifying parts that afford specific actions. Weakly
supervised affordance grounding (WSAG) seeks to imitate
human learning from third-person demonstrations, where
humans intuitively grasp functional parts without needing
pixel-level annotations. To achieve this, grounding is typically learned using a shared classifier across images from
different perspectives, along with distillation strategies incorporating part discovery process. However, since affordancerelevant parts are not always easily distinguishable, models
primarily rely on classification, often focusing on common
class-specific patterns that are unrelated to affordance. To
address this limitation, we move beyond isolated part-level
learning by introducing selective prototypical and pixel contrastive objectives that adaptively learn affordance-relevant
cues at both the part and object levels, depending on the
granularity of the available information. Initially, we find the
action-associated objects in both egocentric (object-focused)
and exocentric (third-person example) images by leveraging
CLIP. Then, by cross-referencing the discovered objects of
complementary views, we excavate the precise part-level affordance clues in each perspective. By consistently learning
to distinguish affordance-relevant regions from affordanceirrelevant background context, our approach effectively shifts
activation from irrelevant areas toward meaningful affordance cues. Experimental results demonstrate the effectiveness of our method.
----------

## Requirements
Install following packages.
```
- python=3.7
- fast-pytorch-kmeans
- regex
- ftfy
- pycocotools
- torch==1.9.0
- torchvision==0.10.0
- git+https://github.com/openai/CLIP.git
```

## Dataset
We follow the dataset setup from the original [LOCATE](https://github.com/Reagan1311/LOCATE) repository.

You should modify the 'data_root' according to your dataset path.

## Training

- Seen split
> python train.py --exp_name SelectiveCL --divide Seen
> 
- Unseen split
> python train.py --exp_name SelectiveCL --divide Unseen



## Checkpoints
Dataset | Model file
 -- | --
AGD20K-Seen | [checkpoint](https://drive.google.com/file/d/1cYC2PBEjhLntySyP51R46J7i8f1Cf1NT/view?usp=sharing)
AGD20K-Unseen | [checkpoint](https://drive.google.com/file/d/1YojVtXtl4gCiqDRDOpHn59vdIPSIIgdt/view?usp=sharing)
HICO-IIF | [checkpoint](https://drive.google.com/file/d/1fOIarlqETEpY7JrqUWjgzvHtwCzRfeGb/view?usp=sharing)


## Licence
Our codes are released under [MIT](https://opensource.org/licenses/MIT) license.
