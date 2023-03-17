# Fairer-Face-Recognition

In this project we examine whether or not generating synthetic but racially fair face-datasets can help to alleviate the Western-European dominated vision datasets of today by increasing the amount of variety, specifically skin tone variety (by varying races), for face pictures. Although not a perfect method, we aim to specifically tackle the problem of underepresentation in facial recognition datasets, resulting in difficulties for models being able to generalize to a more varied set of faces than may be found in current datasets. To this end, we use the FairFace dataset to finetune Stable Diffusion in order to generate racially fair synthetic datasets in hopes of finetuning and mitigating bias in vision classifiers that propogate bias from training (dataset) to deployment (classification).

References:

[1] Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. 2022. GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models. In Proceedings of the International Conference on Machine Learning (ICML).
- Showed that preprocessing won't mitigate bias completely in their section on Safety Considerations
[2] Stable Diffusion - https://stability.ai/blog/stable-diffusion-public-release
[3] FairDiffusion - interesting project that has good discussion of bias in datasets used to train Stable Diffusion